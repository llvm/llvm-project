import gdb
import ompdModule
import itertools
from gdb.FrameDecorator import FrameDecorator
import ompd
from ompd_handles import ompd_task, ompd_parallel, ompd_thread
import traceback
from tempfile import NamedTemporaryFile


class OmpdFrameDecorator(FrameDecorator):
	
	def __init__(self, fobj, curr_task_handle):
		"""Initializes a FrameDecorator with the given GDB Frame object. The global OMPD address space defined in 
		ompd.py is set as well.
		"""
		super(OmpdFrameDecorator, self).__init__(fobj)
		self.addr_space = ompd.addr_space
		self.fobj = None
		if isinstance(fobj, gdb.Frame):
			self.fobj = fobj
		elif isinstance(fobj, FrameDecorator):
			self.fobj = fobj.inferior_frame()
		self.curr_task_handle = curr_task_handle
	
	def function(self):
		"""This appends the name of a frame that is printed with the information whether the task started in the frame 
		is implicit or explicit. The ICVs are evaluated to determine that.
		"""
		name = str(self.fobj.name())
		
		if self.curr_task_handle is None:
			return name
		
		icv_value = ompdModule.call_ompd_get_icv_from_scope(self.curr_task_handle, ompd.icv_map['implicit-task-var'][1], ompd.icv_map['implicit-task-var'][0])
		if icv_value == 0:
			name = '@thread %i: %s "#pragma omp task"' % (gdb.selected_thread().num, name)
		elif icv_value == 1:
			name = '@thread %i: %s "#pragma omp parallel"' % (gdb.selected_thread().num, name)
		else:
			name = '@thread %i: %s' % (gdb.selected_thread().num, name)
		return name

class OmpdFrameDecoratorThread(FrameDecorator):
	
	def __init__(self, fobj):
		"""Initializes a FrameDecorator with the given GDB Frame object."""
		super(OmpdFrameDecoratorThread, self).__init__(fobj)
		if isinstance(fobj, gdb.Frame):
			self.fobj = fobj
		elif isinstance(fobj, FrameDecorator):
			self.fobj = fobj.inferior_frame()
	
	def function(self):
		name = str(self.fobj.name())
		return '@thread %i: %s' % (gdb.selected_thread().num, name)

class FrameFilter():
	
	def __init__(self, addr_space):
		"""Initializes the FrameFilter, registers is in the GDB runtime and saves the given OMPD address space capsule.
		"""
		self.addr_space = addr_space
		self.name = "Filter"
		self.priority = 100
		self.enabled = True
		gdb.frame_filters[self.name] = self
		self.switched_on = False
		self.continue_to_master = False
	
	def set_switch(self, on_off):
		"""Prints output when executing 'ompd bt on' or 'ompd bt off'.
		"""
		self.switched_on = on_off
		if self.switched_on:
			print('Enabled filter for "bt" output successfully.')
		else:
			print('Disabled filter for "bt" output successfully.')
	
	def set_switch_continue(self, on_off):
		"""Prints output when executing 'ompd bt on continued'."
		"""
		self.continue_to_master = on_off
		if self.continue_to_master:
			print('Enabled "bt" mode that continues backtrace on to master thread for worker threads.')
		else:
			print('Disabled "bt" mode that continues onto master thread.')
	
	def get_master_frames_for_worker(self, past_thread_num, latest_sp):
		"""Prints master frames for worker thread with id past_thread_num.
		"""
		gdb.execute('t 1')
		gdb.execute('ompd bt on')
		gdb.execute('bt')
		
		frame = gdb.newest_frame()
		
		while frame.older() is not None:
			print('master frame sp:', str(frame.read_register('sp')))
			yield OmpdFrameDecorator(frame)
			frame = frame.older()
		print('latest sp:', str(latest_sp))
		
		gdb.execute('ompd bt on continued')
		gdb.execute('t %d' % int(past_thread_num))
	
	
	def filter_frames(self, frame_iter):
		"""Iterates through frames and only returns those that are relevant to the application
		being debugged. The OmpdFrameDecorator is applied automatically.
		"""
		curr_thread_num = gdb.selected_thread().num
		is_no_omp_thread = False
		if curr_thread_num in self.addr_space.threads:
			curr_thread_obj = self.addr_space.threads[curr_thread_num]
			self.curr_task = curr_thread_obj.get_current_task()
			self.frames = self.curr_task.get_task_frame()
		else:
			is_no_omp_thread = True
			print('Thread %d is no OpenMP thread, printing all frames:' % curr_thread_num)
		
		stop_iter = False
		for x in frame_iter:
			if is_no_omp_thread:
				yield OmpdFrameDecoratorThread(x)
				continue
			
			if x.inferior_frame().older() is None:
				continue
			if self.curr_task.task_handle is None:
				continue
			
			gdb_sp = int(str(x.inferior_frame().read_register('sp')), 16)
			gdb_sp_next_new = int(str(x.inferior_frame()).split(",")[0].split("=")[1], 16)
			if x.inferior_frame().older():
				gdb_sp_next = int(str(x.inferior_frame().older().read_register('sp')), 16)
			else:
				gdb_sp_next = int(str(x.inferior_frame().read_register('sp')), 16)
			while(1):
				(ompd_enter_frame, ompd_exit_frame) = self.frames
				
				if (ompd_enter_frame != 0 and gdb_sp_next_new < ompd_enter_frame):
					break
				if (ompd_exit_frame != 0 and gdb_sp_next_new < ompd_exit_frame):
					if x.inferior_frame().older().older() and int(str(x.inferior_frame().older().older().read_register('sp')), 16) < ompd_exit_frame:
						if self.continue_to_master:
							yield OmpdFrameDecoratorThread(x)
						else:
							yield OmpdFrameDecorator(x, self.curr_task.task_handle)
					else:
						yield OmpdFrameDecorator(x, self.curr_task.task_handle)
					break
				sched_task_handle = self.curr_task.get_scheduling_task_handle()
				
				if(sched_task_handle is None):
					stop_iter = True
					break
				
				self.curr_task = self.curr_task.get_scheduling_task()
				self.frames = self.curr_task.get_task_frame()
			if stop_iter:
				break
		
		# implementation of "ompd bt continued"
		if self.continue_to_master:
			
			orig_thread = gdb.selected_thread().num
			gdb_threads = dict([(t.num, t) for t in gdb.selected_inferior().threads()])
			
			# iterate through generating tasks until outermost task is reached
			while(1):
				# get OMPD thread id for master thread (systag in GDB output)
				try:
					master_num = self.curr_task.get_task_parallel().get_thread_in_parallel(0).get_thread_id()
				except:
					break
				# search for thread id without the "l" for long via "thread find" and get GDB thread num from output
				hex_str = str(hex(master_num))
				thread_output = gdb.execute('thread find %s' % hex_str[0:len(hex_str)-1], to_string=True).split(" ")
				if thread_output[0] == "No":
					raise ValueError('Master thread num could not be found!')
				gdb_master_num = int(thread_output[1])
				# get task that generated last task of worker thread
				try:
					self.curr_task = self.curr_task.get_task_parallel().get_task_in_parallel(0).get_generating_task()
				except:
					break;
				self.frames = self.curr_task.get_task_frame()
				(enter_frame, exit_frame) = self.frames
				if exit_frame == 0:
					print('outermost generating task was reached')
					break
				
				# save GDB num for worker thread to change back to it later
				worker_thread = gdb.selected_thread().num
				
				# use InferiorThread.switch()
				gdb_threads = dict([(t.num, t) for t in gdb.selected_inferior().threads()])
				gdb_threads[gdb_master_num].switch()
				print('#### switching to thread %i ####' % gdb_master_num)
				
				frame = gdb.newest_frame()
				stop_iter = False
				
				while(not stop_iter):
					if self.curr_task.task_handle is None:
						break
					self.frames = self.curr_task.get_task_frame()
					
					while frame:
						if self.curr_task.task_handle is None:
							break
						
						gdb_sp_next_new = int(str(frame).split(",")[0].split("=")[1], 16)
						
						if frame.older():
							gdb_sp_next = int(str(frame.older().read_register('sp')), 16)
						else:
							gdb_sp_next = int(str(frame.read_register('sp')), 16)
						
						while(1):
							(ompd_enter_frame, ompd_exit_frame) = self.frames
							
							if (ompd_enter_frame != 0 and gdb_sp_next_new < ompd_enter_frame):
								break
							if (ompd_exit_frame == 0 or gdb_sp_next_new < ompd_exit_frame):
								if ompd_exit_frame == 0 or frame.older() and frame.older().older() and int(str(frame.older().older().read_register('sp')), 16) < ompd_exit_frame:
									yield OmpdFrameDecoratorThread(frame)
								else:
									yield OmpdFrameDecorator(frame, self.curr_task.task_handle)
								break
							sched_task_handle = ompdModule.call_ompd_get_scheduling_task_handle(self.curr_task.task_handle)
							
							if(sched_task_handle is None):
								stop_iter = True
								break
							self.curr_task = self.curr_task.get_generating_task()
							self.frames = self.curr_task.get_task_frame()
							
						frame = frame.older()
					break
			
				gdb_threads[worker_thread].switch()
				
			gdb_threads[orig_thread].switch()
	
	
	def filter(self, frame_iter):
		"""Function is called automatically with every 'bt' executed. If switched on, this will only let revelant frames be printed 
		or all frames otherwise. If switched on, a FrameDecorator will be applied to state whether '.ompd_task_entry.' refers to an 
		explicit or implicit task.
		"""
		if self.switched_on:
			return self.filter_frames(frame_iter)
		else:
			return frame_iter

import ompdModule
import gdb
import re
import traceback
from ompd_address_space import ompd_address_space
from ompd_handles import ompd_thread, ompd_task, ompd_parallel
from frame_filter import FrameFilter
from enum import Enum


addr_space = None
ff = None
icv_map = None
ompd_scope_map = {1:'global', 2:'address_space', 3:'thread', 4:'parallel', 5:'implicit_task', 6:'task'}
in_task_function = False

class ompd(gdb.Command):
	def __init__(self):
		super(ompd, self).__init__('ompd',
			gdb.COMMAND_STATUS,
			gdb.COMPLETE_NONE,
			True)

class ompd_init(gdb.Command):
	"""Find and initialize ompd library"""

	# first parameter is command-line input, second parameter is gdb-specific data
	def __init__(self):
		self.__doc__ = 'Find and initialize OMPD library\n usage: ompd init'
		super(ompd_init, self).__init__('ompd init',
						gdb.COMMAND_DATA)

	def invoke(self, arg, from_tty):
		global addr_space
		global ff
		try:
			try:
				print(gdb.newest_frame())
			except:
				gdb.execute("start")
			try:
				lib_list = gdb.parse_and_eval("(char**)ompd_dll_locations")
			except gdb.error:
				raise ValueError("No ompd_dll_locations symbol in execution, make sure to have an OMPD enabled OpenMP runtime");
			
			while(gdb.parse_and_eval("(char**)ompd_dll_locations") == False):
				gdb.execute("tbreak ompd_dll_locations_valid")
				gdb.execute("continue")
			
			lib_list = gdb.parse_and_eval("(char**)ompd_dll_locations")
			
			i = 0
			while(lib_list[i]):
				ret = ompdModule.ompd_open(lib_list[i].string())
				if ret == -1:
					raise ValueError("Handle of OMPD library is not a valid string!")
				if ret == -2:
					print("ret == -2")
					pass # It's ok to fail on dlopen
				if ret == -3:
					print("ret == -3")
					pass # It's ok to fail on dlsym
				if ret < -10:
					raise ValueError("OMPD error code %i!" % (-10 - ret))
					
				if ret > 0:
					print("Loaded OMPD lib successfully!")
					try:
						addr_space = ompd_address_space()
						ff = FrameFilter(addr_space)
					except:
						traceback.print_exc()
					return
				i = i+1
			
			raise ValueError("OMPD library could not be loaded!")
		except:
			traceback.print_exc()

class ompd_threads(gdb.Command):
	"""Register thread ids of current context"""
	def __init__(self):
		self.__doc__ = 'Provide information on threads of current context.\n usage: ompd threads'
		super(ompd_threads, self).__init__('ompd threads',
						gdb.COMMAND_STATUS)
	
	def invoke(self, arg, from_tty):
		global addr_space
		if init_error():
			return
		addr_space.list_threads(True)

def print_parallel_region(curr_parallel, team_size):
	"""Helper function for ompd_parallel_region. To print out the details of the parallel region."""
	for omp_thr in range(team_size):
		thread = curr_parallel.get_thread_in_parallel(omp_thr)
		ompd_state = str(addr_space.states[thread.get_state()[0]])
		ompd_wait_id = thread.get_state()[1]
		task = curr_parallel.get_task_in_parallel(omp_thr)
		task_func_addr = task.get_task_function()
		# Get the function this addr belongs to
		sal = gdb.find_pc_line(task_func_addr)
		block = gdb.block_for_pc(task_func_addr)
		while block and not block.function:
			block = block.superblock
		if omp_thr == 0:
			print('%6d (master) %-37s %ld    0x%lx %-25s %-17s:%d' % \
			(omp_thr, ompd_state, ompd_wait_id, task_func_addr, \
			 block.function.print_name, sal.symtab.filename, sal.line))
		else:
			print('%6d          %-37s %ld    0x%lx %-25s %-17s:%d' % \
			(omp_thr, ompd_state, ompd_wait_id, task_func_addr, \
			 block.function.print_name, sal.symtab.filename, sal.line))

class ompd_parallel_region(gdb.Command):
	"""Parallel Region Details"""
	def __init__(self):
		self.__doc__ = 'Display the details of the current and enclosing parallel regions.\n usage: ompd parallel'
		super(ompd_parallel_region, self).__init__('ompd parallel',
							   gdb.COMMAND_STATUS)

	def invoke(self, arg, from_tty):
		global addr_space
		if init_error():
			return
		if addr_space.icv_map is None:
			addr_space.get_icv_map()
		if addr_space.states is None:
			addr_space.enumerate_states()
		curr_thread_handle = addr_space.get_curr_thread()
		curr_parallel_handle = curr_thread_handle.get_current_parallel_handle()
		curr_parallel = ompd_parallel(curr_parallel_handle)
		while curr_parallel_handle is not None and curr_parallel is not None:
			nest_level = ompdModule.call_ompd_get_icv_from_scope(curr_parallel_handle,\
				     addr_space.icv_map['levels-var'][1], addr_space.icv_map['levels-var'][0])
			if nest_level == 0:
				break
			team_size = ompdModule.call_ompd_get_icv_from_scope(curr_parallel_handle, \
				    addr_space.icv_map['team-size-var'][1], \
				    addr_space.icv_map['team-size-var'][0])
			print ("")
			print ("Parallel Region: Nesting Level %d: Team Size: %d" % (nest_level, team_size))
			print ("================================================")
			print ("")
			print ("OMP Thread Nbr  Thread State                     Wait Id  EntryAddr FuncName                 File:Line");
			print ("======================================================================================================");
			print_parallel_region(curr_parallel, team_size)
			enclosing_parallel = curr_parallel.get_enclosing_parallel()
			enclosing_parallel_handle = curr_parallel.get_enclosing_parallel_handle()
			curr_parallel = enclosing_parallel
			curr_parallel_handle = enclosing_parallel_handle

class ompd_icvs(gdb.Command):
	"""ICVs"""
	def __init__(self):
		self.__doc__ = 'Display the values of the Internal Control Variables.\n usage: ompd icvs'
		super(ompd_icvs, self).__init__('ompd icvs',
						 gdb.COMMAND_STATUS)

	def invoke(self, arg, from_tty):
		global addr_space
		global ompd_scope_map
		if init_error():
			return
		curr_thread_handle = addr_space.get_curr_thread()
		if addr_space.icv_map is None:
			addr_space.get_icv_map()
		print("ICV Name                        Scope                     Value")
		print("===============================================================")

		try:
			for icv_name in addr_space.icv_map:
				scope = addr_space.icv_map[icv_name][1]
				#{1:'global', 2:'address_space', 3:'thread', 4:'parallel', 5:'implicit_task', 6:'task'}
				if scope == 2:
					handle = addr_space.addr_space
				elif scope == 3:
					handle = curr_thread_handle.thread_handle
				elif scope == 4:
					handle = curr_thread_handle.get_current_parallel_handle()
				elif scope == 6:
					handle = curr_thread_handle.get_current_task_handle()
				else:
					raise ValueError("Invalid scope")

				if (icv_name == "nthreads-var" or icv_name == "bind-var"):
					icv_value = ompdModule.call_ompd_get_icv_from_scope(
						    handle, scope, addr_space.icv_map[icv_name][0])
					if icv_value is None:
						icv_string = ompdModule.call_ompd_get_icv_string_from_scope( \
							     handle, scope, addr_space.icv_map[icv_name][0])
						print('%-31s %-26s %s' % (icv_name, ompd_scope_map[scope], icv_string))
					else:
						print('%-31s %-26s %d' % (icv_name, ompd_scope_map[scope], icv_value))

				elif (icv_name == "affinity-format-var" or icv_name == "run-sched-var" or \
                                         icv_name == "tool-libraries-var" or icv_name == "tool-verbose-init-var"):
					icv_string = ompdModule.call_ompd_get_icv_string_from_scope( \
						     handle, scope, addr_space.icv_map[icv_name][0])
					print('%-31s %-26s %s' % (icv_name, ompd_scope_map[scope], icv_string))
				else:
					icv_value = ompdModule.call_ompd_get_icv_from_scope(handle, \
						    scope, addr_space.icv_map[icv_name][0])
					print('%-31s %-26s %d' % (icv_name, ompd_scope_map[scope], icv_value))
		except:
		       traceback.print_exc()

def curr_thread():
	"""Helper function for ompd_step. Returns the thread object for the current thread number."""
	global addr_space
	if addr_space is not None:
		return addr_space.threads[int(gdb.selected_thread().num)]
	return None

class ompd_test(gdb.Command):
	"""Test area"""
	def __init__(self):
		self.__doc__ = 'Test functionalities for correctness\n usage: ompd test'
		super(ompd_test, self).__init__('ompd test',
						gdb.COMMAND_OBSCURE)
	
	def invoke(self, arg, from_tty):
		global addr_space
		if init_error():
			return
		# get task function for current task of current thread
		try:
			current_thread = int(gdb.selected_thread().num)
			current_thread_obj = addr_space.threads[current_thread]
			task_function = current_thread_obj.get_current_task().get_task_function()
			print("bt value:", int("0x0000000000400b6c",0))
			print("get_task_function value:", task_function)

			# get task function of implicit task in current parallel region for current thread
			current_parallel_obj = current_thread_obj.get_current_parallel()
			task_in_parallel = current_parallel_obj.get_task_in_parallel(current_thread)
			task_function_in_parallel = task_in_parallel.get_task_function()
			print("task_function_in_parallel:", task_function_in_parallel)
		except:
			print('Task function value not found for this thread')

class ompdtestapi (gdb.Command):
	""" To test API's return code """
	def __init__(self):
		self.__doc__ = 'Test OMPD tool Interface APIs.\nUsage: ompdtestapi <api name>'
		super (ompdtestapi, self).__init__('ompdtestapi', gdb.COMMAND_OBSCURE)

	def invoke (self, arg, from_tty):
		global addr_space
		if init_error():
			print ("Error in Initialization.");
			return
		if not arg:
			print ("No API provided to test, eg: ompdtestapi ompd_initialize")

		if arg == "ompd_get_thread_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			ompdModule.test_ompd_get_thread_handle(addr_handle, threadId)
		elif arg == "ompd_get_curr_parallel_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			ompdModule.test_ompd_get_curr_parallel_handle(thread_handle)
		elif arg == "ompd_get_thread_in_parallel":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			ompdModule.test_ompd_get_thread_in_parallel(parallel_handle)
		elif arg == "ompd_thread_handle_compare":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			thread_handle1 = ompdModule.call_ompd_get_thread_in_parallel(parallel_handle, 1);
			thread_handle2 = ompdModule.call_ompd_get_thread_in_parallel(parallel_handle, 2);
			ompdModule.test_ompd_thread_handle_compare(thread_handle1, thread_handle1)
			ompdModule.test_ompd_thread_handle_compare(thread_handle1, thread_handle2)
		elif arg == "ompd_get_thread_id":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			ompdModule.test_ompd_get_thread_id(thread_handle)
		elif arg == "ompd_rel_thread_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			ompdModule.test_ompd_rel_thread_handle(thread_handle)
		elif arg == "ompd_get_enclosing_parallel_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			ompdModule.test_ompd_get_enclosing_parallel_handle(parallel_handle)
		elif arg == "ompd_parallel_handle_compare":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle1 = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			parallel_handle2 = ompdModule.call_ompd_get_enclosing_parallel_handle(parallel_handle1)
			ompdModule.test_ompd_parallel_handle_compare(parallel_handle1, parallel_handle1)
			ompdModule.test_ompd_parallel_handle_compare(parallel_handle1, parallel_handle2)
		elif arg == "ompd_rel_parallel_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			ompdModule.test_ompd_rel_parallel_handle(parallel_handle)
		elif arg == "ompd_initialize":
			ompdModule.test_ompd_initialize()
		elif arg == "ompd_get_api_version":
			ompdModule.test_ompd_get_api_version()
		elif arg == "ompd_get_version_string":
			ompdModule.test_ompd_get_version_string()
		elif arg == "ompd_finalize":
			ompdModule.test_ompd_finalize()
		elif arg == "ompd_process_initialize":
			ompdModule.call_ompd_initialize()
			ompdModule.test_ompd_process_initialize()
		elif arg == "ompd_device_initialize":
			ompdModule.test_ompd_device_initialize()
		elif arg == "ompd_rel_address_space_handle":
			ompdModule.test_ompd_rel_address_space_handle()
		elif arg == "ompd_get_omp_version":
			addr_handle = addr_space.addr_space
			ompdModule.test_ompd_get_omp_version(addr_handle)
		elif arg == "ompd_get_omp_version_string":
			addr_handle = addr_space.addr_space
			ompdModule.test_ompd_get_omp_version_string(addr_handle)
		elif arg == "ompd_get_curr_task_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			ompdModule.test_ompd_get_curr_task_handle(thread_handle)
		elif arg == "ompd_get_task_parallel_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_get_task_parallel_handle(task_handle)
		elif arg == "ompd_get_generating_task_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_get_generating_task_handle(task_handle)
		elif arg == "ompd_get_scheduling_task_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_get_scheduling_task_handle(task_handle)
		elif arg == "ompd_get_task_in_parallel":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			ompdModule.test_ompd_get_task_in_parallel(parallel_handle)
		elif arg == "ompd_rel_task_handle":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_rel_task_handle(task_handle)
		elif arg == "ompd_task_handle_compare":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle1 = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			task_handle2 = ompdModule.call_ompd_get_generating_task_handle(task_handle1)
			ompdModule.test_ompd_task_handle_compare(task_handle1, task_handle2)
			ompdModule.test_ompd_task_handle_compare(task_handle2, task_handle1)
		elif arg == "ompd_get_task_function":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_get_task_function(task_handle)
		elif arg == "ompd_get_task_frame":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)
			ompdModule.test_ompd_get_task_frame(task_handle)
		elif arg == "ompd_get_state":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			ompdModule.test_ompd_get_state(thread_handle)
		elif arg == "ompd_get_display_control_vars":
			addr_handle = addr_space.addr_space
			ompdModule.test_ompd_get_display_control_vars(addr_handle)
		elif arg == "ompd_rel_display_control_vars":
			ompdModule.test_ompd_rel_display_control_vars()
		elif arg == "ompd_enumerate_icvs":
			addr_handle = addr_space.addr_space
			ompdModule.test_ompd_enumerate_icvs(addr_handle)
		elif arg== "ompd_get_icv_from_scope":
			addr_handle = addr_space.addr_space
			threadId = gdb.selected_thread().ptid[1]
			thread_handle = ompdModule.get_thread_handle(threadId, addr_handle)
			parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(thread_handle)
			task_handle = ompdModule.call_ompd_get_curr_task_handle(thread_handle)	
			ompdModule.test_ompd_get_icv_from_scope_with_addr_handle(addr_handle)
			ompdModule.test_ompd_get_icv_from_scope_with_thread_handle(thread_handle)
			ompdModule.test_ompd_get_icv_from_scope_with_parallel_handle(parallel_handle)
			ompdModule.test_ompd_get_icv_from_scope_with_task_handle(task_handle)
		elif arg == "ompd_get_icv_string_from_scope":
			addr_handle = addr_space.addr_space
			ompdModule.test_ompd_get_icv_string_from_scope(addr_handle)
		elif arg == "ompd_get_tool_data":
			ompdModule.test_ompd_get_tool_data()
		elif arg == "ompd_enumerate_states":
			ompdModule.test_ompd_enumerate_states()
		else:
			print ("Invalid API.")



class ompd_bt(gdb.Command):
	"""Turn filter for 'bt' on/off for output to only contain frames relevant to the application or all frames."""
	def __init__(self):
		self.__doc__ = 'Turn filter for "bt" output on or off. Specify "on continued" option to trace worker threads back to master threads.\n usage: ompd bt on|on continued|off'
		super(ompd_bt, self).__init__('ompd bt',
					gdb.COMMAND_STACK)
	
	def invoke(self, arg, from_tty):
		global ff
		global addr_space
		global icv_map
		global ompd_scope_map
		if init_error():
			return
		if icv_map is None:
			icv_map = {}
			current = 0
			more = 1
			while more > 0:
				tup = ompdModule.call_ompd_enumerate_icvs(addr_space.addr_space, current)
				(current, next_icv, next_scope, more) = tup
				icv_map[next_icv] = (current, next_scope, ompd_scope_map[next_scope])
			print('Initialized ICV map successfully for filtering "bt".')
		
		arg_list = gdb.string_to_argv(arg)
		if len(arg_list) == 0:
			print('When calling "ompd bt", you must either specify "on", "on continued" or "off". Check "help ompd".')
		elif len(arg_list) == 1 and arg_list[0] == 'on':
			addr_space.list_threads(False)
			ff.set_switch(True)
			ff.set_switch_continue(False)
		elif arg_list[0] == 'on' and arg_list[1] == 'continued':
			ff.set_switch(True)
			ff.set_switch_continue(True)
		elif len(arg_list) == 1 and arg_list[0] == 'off':
			ff.set_switch(False)
			ff.set_switch_continue(False)
		else:
			print('When calling "ompd bt", you must either specify "on", "on continued" or "off". Check "help ompd".')

# TODO: remove
class ompd_taskframes(gdb.Command):
	"""Prints task handles for relevant task frames. Meant for debugging."""
	def __init__(self):
		self.__doc__ = 'Prints list of tasks.\nUsage: ompd taskframes'
		super(ompd_taskframes, self).__init__('ompd taskframes',
					gdb.COMMAND_STACK)
	
	def invoke(self, arg, from_tty):
		global addr_space
		if init_error():
			return
		frame = gdb.newest_frame()
		while(frame):
			print (frame.read_register('sp'))
			frame = frame.older()
		curr_task_handle = None
		if(addr_space.threads and addr_space.threads.get(gdb.selected_thread().num)):
			curr_thread_handle = curr_thread().thread_handle
			curr_task_handle = ompdModule.call_ompd_get_curr_task_handle(curr_thread_handle)
		if(not curr_task_handle):
			return None
		prev_frames = None
		try:
			while(1):
				frames_with_flags = ompdModule.call_ompd_get_task_frame(curr_task_handle)
				frames = (frames_with_flags[0], frames_with_flags[3])
				if(prev_frames == frames):
					break
				if(not isinstance(frames,tuple)):
					break
				(ompd_enter_frame, ompd_exit_frame) = frames
				print(hex(ompd_enter_frame), hex(ompd_exit_frame))
				curr_task_handle = ompdModule.call_ompd_get_scheduling_task_handle(curr_task_handle)
				prev_frames = frames
				if(not curr_task_handle):
					break
		except:
			traceback.print_exc()

def print_and_exec(string):
	"""Helper function for ompd_step. Executes the given command in GDB and prints it."""
	print(string)
	gdb.execute(string)

class TempFrameFunctionBp(gdb.Breakpoint):
	"""Helper class for ompd_step. Defines stop function for breakpoint on frame function."""
	def stop(self):
		global in_task_function
		in_task_function = True
		self.enabled = False

class ompd_step(gdb.Command):
	"""Executes 'step' and skips frames irrelevant to the application / the ones without debug information."""
	def __init__(self):
		self.__doc__ = 'Executes "step" and skips runtime frames as much as possible.'
		super(ompd_step, self).__init__('ompd step', gdb.COMMAND_STACK)
	
	class TaskBeginBp(gdb.Breakpoint):
		"""Helper class. Defines stop function for breakpoint ompd_bp_task_begin."""
		def stop(self):
			try:
				code_line = curr_thread().get_current_task().get_task_function()
				frame_fct_bp = TempFrameFunctionBp(('*%i' % code_line), temporary=True, internal=True)
				frame_fct_bp.thread = self.thread
				return False
			except:
				return False
	
	def invoke(self, arg, from_tty):
		global in_task_function
		if init_error():
			return
		tbp = self.TaskBeginBp('ompd_bp_task_begin', temporary=True, internal=True)
		tbp.thread = int(gdb.selected_thread().num)
		print_and_exec('step')
		while gdb.selected_frame().find_sal().symtab is None:
			if not in_task_function:
				print_and_exec('finish')
			else:
				print_and_exec('si')

def init_error():
	global addr_space
	if (gdb.selected_thread() is None) or (addr_space is None) or (not addr_space):
		print("Run 'ompd init' before running any of the ompd commands")
		return True
	return False

def main():
	ompd()
	ompd_init()
	ompd_threads()
	ompd_icvs()
	ompd_parallel_region()
	ompd_test()
	ompdtestapi()
	ompd_taskframes()
	ompd_bt()
	ompd_step()

if __name__ == "__main__":
	try:
		main()
	except:
		traceback.print_exc()

# NOTE: test code using:
# OMP_NUM_THREADS=... gdb a.out -x ../../projects/gdb_plugin/gdb-ompd/__init__.py
# ompd init
# ompd threads

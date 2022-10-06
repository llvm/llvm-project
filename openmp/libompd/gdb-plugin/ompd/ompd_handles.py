import ompdModule
import imp

class ompd_parallel(object):
	
	def __init__(self, parallel_handle):
		""" Initializes an ompd_parallel object with the pointer
		to a handle of a parallel region."""
		self.parallel_handle = parallel_handle
		self.threads = {}
		self.itasks = {}
		self.enclosing_parallel_handle = None
		self.enclosing_parallel = False
		self.task_handle = None
	
	def get_thread_in_parallel(self, thread_num):
		"""Obtains thread handles for the threads associated with the
		parallel region specified by parallel_handle."""
		if not thread_num in self.threads:
			thread_handle = ompdModule.call_ompd_get_thread_in_parallel(self.parallel_handle, thread_num)
			self.threads[thread_num] = ompd_thread(thread_handle)
		return self.threads[thread_num]
	
	def get_enclosing_parallel_handle(self):
		"""Obtains a parallel handle for the parallel region enclosing
		the parallel region specified by parallel_handle."""
		if not self.enclosing_parallel_handle:
			self.enclosing_parallel_handle = ompdModule.call_ompd_get_enclosing_parallel_handle(self.parallel_handle)
		return self.enclosing_parallel_handle
	
	def get_enclosing_parallel(self):
		if not self.enclosing_parallel:
			self.enclosing_parallel = ompd_parallel(self.get_enclosing_parallel_handle())
		return self.enclosing_parallel
	
	def get_task_in_parallel(self, thread_num):
		"""Obtains handles for the implicit tasks associated with the
		parallel region specified by parallel_handle."""
		if not thread_num in self.itasks:
			task_handle = ompdModule.call_ompd_get_task_in_parallel(self.parallel_handle, thread_num)
			self.itasks[thread_num] = ompd_task(task_handle)
		return self.itasks[thread_num]
	
	def __del__(self):
		"""Releases the parallel handle."""
		pass # let capsule destructors do the job

class ompd_task(object):
	
	def __init__(self, task_handle):
		"""Initializes a new ompd_task_handle object and sets the attribute
		to the task handle specified."""
		self.task_handle = task_handle
		self.task_parallel_handle = False
		self.generating_task_handle = False
		self.scheduling_task_handle = False
		self.task_parallel = False
		self.generating_task = False
		self.scheduling_task = False
		self.task_frames = None
		self.task_frame_flags = None
	
	def get_task_parallel_handle(self):
		"""Obtains a task parallel handle for the parallel region enclosing
		the task region specified."""
		if not self.task_parallel_handle:
			self.task_parallel_handle = ompdModule.call_ompd_get_task_parallel_handle(self.task_handle)
		return self.task_parallel_handle
	
	def get_task_parallel(self):
		if not self.task_parallel:
			self.task_parallel = ompd_parallel(self.get_task_parallel_handle())
		return self.task_parallel
	
	def get_generating_task_handle(self):
		"""Obtains the task handle for the task that created the task specified
		by the task handle."""
		if not self.generating_task_handle:
			self.generating_task_handle = ompdModule.call_ompd_get_generating_task_handle(self.task_handle)
		return self.generating_task_handle
	
	def get_generating_task(self):
		if not self.generating_task:
			self.generating_task = ompd_task(ompdModule.call_ompd_get_generating_task_handle(self.task_handle))
		return self.generating_task
	
	def get_scheduling_task_handle(self):
		"""Obtains the task handle for the task that scheduled the task specified."""
		if not self.scheduling_task_handle:
			self.scheduling_task_handle = ompdModule.call_ompd_get_scheduling_task_handle(self.task_handle)
		return self.scheduling_task_handle
	
	def get_scheduling_task(self):
		"""Returns ompd_task object for the task that scheduled the current task."""
		if not self.scheduling_task:
			self.scheduling_task = ompd_task(self.get_scheduling_task_handle())
		return self.scheduling_task

	def get_task_function(self):
		"""Returns long with address of function entry point."""
		return ompdModule.call_ompd_get_task_function(self.task_handle)
	
	def get_task_frame_with_flags(self):
		"""Returns enter frame address and flag, exit frame address and flag for current task handle."""
		if self.task_frames is None or self.task_frame_flags is None:
			ret_value = ompdModule.call_ompd_get_task_frame(self.task_handle)
			if isinstance(ret_value, tuple):
				self.task_frames = (ret_value[0], ret_value[2])
				self.task_frame_flags = (ret_value[1], ret_value[3])
			else:
				return ret_value
		return (self.task_frames[0], self.task_frame_flags[0], self.task_frames[1], self.task_frame_flags[1])
	
	def get_task_frame(self):
		"""Returns enter and exit frame address for current task handle."""
		if self.task_frames is None:
			ret_value = ompdModule.call_ompd_get_task_frame(self.task_handle)
			if isinstance(ret_value, tuple):
				self.task_frames = (ret_value[0], ret_value[2])
			else:
				return ret_value
		return self.task_frames
	
	def __del__(self):
		"""Releases the task handle."""
		pass # let capsule destructors do the job


class ompd_thread(object):
	
	def __init__(self, thread_handle):
		"""Initializes an ompd_thread with the data received from
		GDB."""
		self.thread_handle = thread_handle
		self.parallel_handle = None
		self.task_handle = None
		self.current_task = False
		self.current_parallel = False
		self.thread_id = False
	
	def get_current_parallel_handle(self):
		"""Obtains the parallel handle for the parallel region associated with
		the given thread handle."""
		#TODO: invalidate thread objects based on `gdb.event.cont`. This should invalidate all internal state.
		self.parallel_handle = ompdModule.call_ompd_get_curr_parallel_handle(self.thread_handle)
		return self.parallel_handle
	
	def get_current_parallel(self):
		"""Returns parallel object for parallel handle of the parallel region 
		associated with the current thread handle."""
		if not self.current_parallel:
			self.current_parallel = ompd_parallel(self.get_current_parallel_handle())
		return self.current_parallel
		
	def get_current_task_handle(self):
		"""Obtains the task handle for the current task region of the
		given thread."""
		return ompdModule.call_ompd_get_curr_task_handle(self.thread_handle)

	def get_thread_id(self):
		"""Obtains the ID for the given thread."""
		if not self.thread_id:
			self.thread_id = ompdModule.call_ompd_get_thread_id(self.thread_handle)
		return self.thread_id

	def get_current_task(self):
		"""Returns task object for task handle of the current task region."""
		return ompd_task(self.get_current_task_handle())
	
	def get_state(self):
		"""Returns tuple with OMPD state (long) and wait_id, in case the thread is in a 
		waiting state. Helper function for 'ompd threads' command."""
		(state, wait_id) = ompdModule.call_ompd_get_state(self.thread_handle)
		return (state, wait_id)
	
	def __del__(self):
		"""Releases the given thread handle."""
		pass # let capsule destructors do the job

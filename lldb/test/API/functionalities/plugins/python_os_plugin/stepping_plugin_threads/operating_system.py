#!/usr/bin/env python

import lldb
import struct


class OperatingSystemPlugIn(object):
    """Class that provides a OS plugin that along with the particular code in main.cpp
       emulates the following scenario:
             a) We stop in an OS Plugin created thread - which should be thread index 1
             b) We step-out from that thread
             c) We hit a breakpoint in another thread, and DON'T produce the OS Plugin thread.
             d) We continue, and when we hit the step out breakpoint, we again produce the same
                OS Plugin thread.
             main.cpp sets values into the global variable g_value, which we use to tell the OS
             plugin whether to produce the OS plugin thread or not.
             Since we are always producing an OS plugin thread with a backing thread, we don't
             need to implement get_register_info or get_register_data.
    """

    def __init__(self, process):
        '''Initialization needs a valid.SBProcess object.

        This plug-in will get created after a live process is valid and has stopped for the
        first time.'''
        print("Plugin initialized.")
        self.process = None
        self.start_stop_id = 0
        self.g_value = lldb.SBValue()
        
        if isinstance(process, lldb.SBProcess) and process.IsValid():
            self.process = process
            self.g_value = process.GetTarget().FindFirstGlobalVariable("g_value")
            if not self.g_value.IsValid():
                print("Could not find g_value")
            
    def create_thread(self, tid, context):
        print("Called create thread with tid: ", tid)
        return None

    def get_thread_info(self):
        g_value = self.g_value.GetValueAsUnsigned()
        print("Called get_thread_info: g_value: %d"%(g_value))
        if g_value == 0 or g_value == 2:
            return [{'tid': 0x111111111,
                             'name': 'one',
                             'queue': 'queue1',
                             'state': 'stopped',
                             'stop_reason': 'breakpoint',
                             'core' : 1 }]
        else:
            return []

    def get_register_info(self):
        print ("called get_register_info")
        return None

    
    def get_register_data(self, tid):
        print("Get register data called for tid: %d"%(tid))
        return None


from __future__ import print_function
import ompdModule
from ompd_handles import ompd_thread, ompd_task, ompd_parallel
import gdb
import sys
import traceback
from enum import Enum


class ompd_scope(Enum):
    ompd_scope_global = 1
    ompd_scope_address_space = 2
    ompd_scope_thread = 3
    ompd_scope_parallel = 4
    ompd_scope_implicit_task = 5
    ompd_scope_task = 6


class ompd_address_space(object):
    def __init__(self):
        """Initializes an ompd_address_space object by calling ompd_initialize
        in ompdModule.c
        """
        self.addr_space = ompdModule.call_ompd_initialize()
        # maps thread_num (thread id given by gdb) to ompd_thread object with thread handle
        self.threads = {}
        self.states = None
        self.icv_map = None
        self.ompd_tool_test_bp = None
        self.scope_map = {
            1: "global",
            2: "address_space",
            3: "thread",
            4: "parallel",
            5: "implicit_task",
            6: "task",
        }
        self.sched_map = {1: "static", 2: "dynamic", 3: "guided", 4: "auto"}
        gdb.events.stop.connect(self.handle_stop_event)
        self.new_thread_breakpoint = gdb.Breakpoint(
            "ompd_bp_thread_begin", internal=True
        )
        tool_break_symbol = gdb.lookup_global_symbol("ompd_tool_break")
        if tool_break_symbol is not None:
            self.ompd_tool_test_bp = gdb.Breakpoint("ompd_tool_break", internal=True)

    def handle_stop_event(self, event):
        """Sets a breakpoint at different events, e.g. when a new OpenMP
        thread is created.
        """
        if isinstance(event, gdb.BreakpointEvent):
            # check if breakpoint has already been hit
            if self.new_thread_breakpoint in event.breakpoints:
                self.add_thread()
                gdb.execute("continue")
                return
            elif (
                self.ompd_tool_test_bp is not None
                and self.ompd_tool_test_bp in event.breakpoints
            ):
                try:
                    self.compare_ompt_data()
                    gdb.execute("continue")
                except ():
                    traceback.print_exc()
        elif isinstance(event, gdb.SignalEvent):
            # TODO: what do we need to do on SIGNALS?
            pass
        else:
            # TODO: probably not possible?
            pass

    def get_icv_map(self):
        """Fills ICV map."""
        self.icv_map = {}
        current = 0
        more = 1
        while more > 0:
            tup = ompdModule.call_ompd_enumerate_icvs(self.addr_space, current)
            (current, next_icv, next_scope, more) = tup
            self.icv_map[next_icv] = (current, next_scope, self.scope_map[next_scope])
        print("Initialized ICV map successfully for checking OMP API values.")

    def compare_ompt_data(self):
        """Compares OMPT tool data about parallel region to data returned by OMPD functions."""
        # make sure all threads and states are set
        self.list_threads(False)

        thread_id = gdb.selected_thread().ptid[1]
        curr_thread = self.get_curr_thread()

        # check if current thread is LWP thread; return if "ompd_rc_unavailable"
        thread_handle = ompdModule.get_thread_handle(thread_id, self.addr_space)
        if thread_handle == -1:
            print("Skipping OMPT-OMPD checks for non-LWP thread.")
            return

        print("Comparing OMPT data to OMPD data...")
        field_names = [i.name for i in gdb.parse_and_eval("thread_data").type.fields()]
        thread_data = gdb.parse_and_eval("thread_data")

        if self.icv_map is None:
            self.get_icv_map()

        # compare state values
        if "ompt_state" in field_names:
            if self.states is None:
                self.enumerate_states()
            ompt_state = str(thread_data["ompt_state"])
            ompd_state = str(self.states[curr_thread.get_state()[0]])
            if ompt_state != ompd_state:
                print(
                    "OMPT-OMPD mismatch: ompt_state (%s) does not match OMPD state (%s)!"
                    % (ompt_state, ompd_state)
                )

        # compare wait_id values
        if "ompt_wait_id" in field_names:
            ompt_wait_id = thread_data["ompt_wait_id"]
            ompd_wait_id = curr_thread.get_state()[1]
            if ompt_wait_id != ompd_wait_id:
                print(
                    "OMPT-OMPD mismatch: ompt_wait_id (%d) does not match OMPD wait id (%d)!"
                    % (ompt_wait_id, ompd_wait_id)
                )

        # compare thread id
        if "omp_thread_num" in field_names and "thread-num-var" in self.icv_map:
            ompt_thread_num = thread_data["omp_thread_num"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.thread_handle,
                self.icv_map["thread-num-var"][1],
                self.icv_map["thread-num-var"][0],
            )
            if ompt_thread_num != icv_value:
                print(
                    "OMPT-OMPD mismatch: omp_thread_num (%d) does not match OMPD thread num according to ICVs (%d)!"
                    % (ompt_thread_num, icv_value)
                )

        # compare thread data
        if "ompt_thread_data" in field_names:
            ompt_thread_data = thread_data["ompt_thread_data"].dereference()["value"]
            ompd_value = ompdModule.call_ompd_get_tool_data(
                3, curr_thread.thread_handle
            )[0]
            if ompt_thread_data != ompd_value:
                print(
                    "OMPT-OMPD mismatch: value of ompt_thread_data (%d) does not match that of OMPD data union (%d)!"
                    % (ompt_thread_data, ompd_value)
                )

        # compare number of threads
        if "omp_num_threads" in field_names and "team-size-var" in self.icv_map:
            ompt_num_threads = thread_data["omp_num_threads"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_parallel_handle(),
                self.icv_map["team-size-var"][1],
                self.icv_map["team-size-var"][0],
            )
            if ompt_num_threads != icv_value:
                print(
                    "OMPT-OMPD mismatch: omp_num_threads (%d) does not match OMPD num threads according to ICVs (%d)!"
                    % (ompt_num_threads, icv_value)
                )

        # compare omp level
        if "omp_level" in field_names and "levels-var" in self.icv_map:
            ompt_levels = thread_data["omp_level"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_parallel_handle(),
                self.icv_map["levels-var"][1],
                self.icv_map["levels-var"][0],
            )
            if ompt_levels != icv_value:
                print(
                    "OMPT-OMPD mismatch: omp_level (%d) does not match OMPD levels according to ICVs (%d)!"
                    % (ompt_levels, icv_value)
                )

        # compare active level
        if "omp_active_level" in field_names and "active-levels-var" in self.icv_map:
            ompt_active_levels = thread_data["omp_active_level"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_parallel_handle(),
                self.icv_map["active-levels-var"][1],
                self.icv_map["active-levels-var"][0],
            )
            if ompt_active_levels != icv_value:
                print(
                    "OMPT-OMPD mismatch: active levels (%d) do not match active levels according to ICVs (%d)!"
                    % (ompt_active_levels, icv_value)
                )

        # compare parallel data
        if "ompt_parallel_data" in field_names:
            ompt_parallel_data = thread_data["ompt_parallel_data"].dereference()[
                "value"
            ]
            current_parallel_handle = curr_thread.get_current_parallel_handle()
            ompd_value = ompdModule.call_ompd_get_tool_data(4, current_parallel_handle)[
                0
            ]
            if ompt_parallel_data != ompd_value:
                print(
                    "OMPT-OMPD mismatch: value of ompt_parallel_data (%d) does not match that of OMPD data union (%d)!"
                    % (ompt_parallel_data, ompd_value)
                )

        # compare max threads
        if "omp_max_threads" in field_names and "nthreads-var" in self.icv_map:
            ompt_max_threads = thread_data["omp_max_threads"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.thread_handle,
                self.icv_map["nthreads-var"][1],
                self.icv_map["nthreads-var"][0],
            )
            if icv_value is None:
                icv_string = ompdModule.call_ompd_get_icv_string_from_scope(
                    curr_thread.thread_handle,
                    self.icv_map["nthreads-var"][1],
                    self.icv_map["nthreads-var"][0],
                )
                if icv_string is None:
                    print(
                        "OMPT-OMPD mismatch: omp_max_threads (%d) does not match OMPD thread limit according to ICVs (None Object)"
                        % (ompt_max_threads)
                    )
                else:
                    if ompt_max_threads != int(icv_string.split(",")[0]):
                        print(
                            "OMPT-OMPD mismatch: omp_max_threads (%d) does not match OMPD thread limit according to ICVs (%d)!"
                            % (ompt_max_threads, int(icv_string.split(",")[0]))
                        )
            else:
                if ompt_max_threads != icv_value:
                    print(
                        "OMPT-OMPD mismatch: omp_max_threads (%d) does not match OMPD thread limit according to ICVs (%d)!"
                        % (ompt_max_threads, icv_value)
                    )

        # compare omp_parallel
        # NOTE: omp_parallel = true if active-levels-var > 0
        if "omp_parallel" in field_names:
            ompt_parallel = thread_data["omp_parallel"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_parallel_handle(),
                self.icv_map["active-levels-var"][1],
                self.icv_map["active-levels-var"][0],
            )
            if (
                ompt_parallel == 1
                and icv_value <= 0
                or ompt_parallel == 0
                and icv_value > 0
            ):
                print(
                    "OMPT-OMPD mismatch: ompt_parallel (%d) does not match OMPD parallel according to ICVs (%d)!"
                    % (ompt_parallel, icv_value)
                )

        # compare omp_final
        if "omp_final" in field_names and "final-task-var" in self.icv_map:
            ompt_final = thread_data["omp_final"]
            current_task_handle = curr_thread.get_current_task_handle()
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                current_task_handle,
                self.icv_map["final-task-var"][1],
                self.icv_map["final-task-var"][0],
            )
            if icv_value != ompt_final:
                print(
                    "OMPT-OMPD mismatch: omp_final (%d) does not match OMPD final according to ICVs (%d)!"
                    % (ompt_final, icv_value)
                )

        # compare omp_dynamic
        if "omp_dynamic" in field_names and "dyn-var" in self.icv_map:
            ompt_dynamic = thread_data["omp_dynamic"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.thread_handle,
                self.icv_map["dyn-var"][1],
                self.icv_map["dyn-var"][0],
            )
            if icv_value != ompt_dynamic:
                print(
                    "OMPT-OMPD mismatch: omp_dynamic (%d) does not match OMPD dynamic according to ICVs (%d)!"
                    % (ompt_dynamic, icv_value)
                )

        # compare omp_max_active_levels
        if (
            "omp_max_active_levels" in field_names
            and "max-active-levels-var" in self.icv_map
        ):
            ompt_max_active_levels = thread_data["omp_max_active_levels"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_task_handle(),
                self.icv_map["max-active-levels-var"][1],
                self.icv_map["max-active-levels-var"][0],
            )
            if ompt_max_active_levels != icv_value:
                print(
                    "OMPT-OMPD mismatch: omp_max_active_levels (%d) does not match OMPD max active levels (%d)!"
                    % (ompt_max_active_levels, icv_value)
                )

                # compare omp_kind: TODO: Add the test for monotonic/nonmonotonic modifier
        if "omp_kind" in field_names and "run-sched-var" in self.icv_map:
            ompt_sched_kind = thread_data["omp_kind"]
            icv_value = ompdModule.call_ompd_get_icv_string_from_scope(
                curr_thread.get_current_task_handle(),
                self.icv_map["run-sched-var"][1],
                self.icv_map["run-sched-var"][0],
            )
            ompd_sched_kind = icv_value.split(",")[0]
            if self.sched_map.get(int(ompt_sched_kind)) != ompd_sched_kind:
                print(
                    "OMPT-OMPD mismatch: omp_kind kind (%s) does not match OMPD schedule kind according to ICVs (%s)!"
                    % (self.sched_map.get(int(ompt_sched_kind)), ompd_sched_kind)
                )

        # compare omp_modifier
        if "omp_modifier" in field_names and "run-sched-var" in self.icv_map:
            ompt_sched_mod = thread_data["omp_modifier"]
            icv_value = ompdModule.call_ompd_get_icv_string_from_scope(
                curr_thread.get_current_task_handle(),
                self.icv_map["run-sched-var"][1],
                self.icv_map["run-sched-var"][0],
            )
            token = icv_value.split(",")[1]
            if token is not None:
                ompd_sched_mod = int(token)
            else:
                ompd_sched_mod = 0
            if ompt_sched_mod != ompd_sched_mod:
                print(
                    "OMPT-OMPD mismatch: omp_kind modifier does not match OMPD schedule modifier according to ICVs!"
                )

        # compare omp_proc_bind
        if "omp_proc_bind" in field_names and "bind-var" in self.icv_map:
            ompt_proc_bind = thread_data["omp_proc_bind"]
            icv_value = ompdModule.call_ompd_get_icv_from_scope(
                curr_thread.get_current_task_handle(),
                self.icv_map["bind-var"][1],
                self.icv_map["bind-var"][0],
            )
            if icv_value is None:
                icv_string = ompdModule.call_ompd_get_icv_string_from_scope(
                    curr_thread.get_current_task_handle(),
                    self.icv_map["bind-var"][1],
                    self.icv_map["bind-var"][0],
                )
                if icv_string is None:
                    print(
                        "OMPT-OMPD mismatch: omp_proc_bind (%d) does not match OMPD proc bind according to ICVs (None Object)"
                        % (ompt_proc_bind)
                    )
                else:
                    if ompt_proc_bind != int(icv_string.split(",")[0]):
                        print(
                            "OMPT-OMPD mismatch: omp_proc_bind (%d) does not match OMPD proc bind according to ICVs (%d)!"
                            % (ompt_proc_bind, int(icv_string.split(",")[0]))
                        )
            else:
                if ompt_proc_bind != icv_value:
                    print(
                        "OMPT-OMPD mismatch: omp_proc_bind (%d) does not match OMPD proc bind according to ICVs (%d)!"
                        % (ompt_proc_bind, icv_value)
                    )

        # compare enter and exit frames
        if "ompt_frame_list" in field_names:
            ompt_task_frame_dict = thread_data["ompt_frame_list"].dereference()
            ompt_task_frames = (
                int(ompt_task_frame_dict["enter_frame"].cast(gdb.lookup_type("long"))),
                int(ompt_task_frame_dict["exit_frame"].cast(gdb.lookup_type("long"))),
            )
            current_task = curr_thread.get_current_task()
            ompd_task_frames = current_task.get_task_frame()
            if ompt_task_frames != ompd_task_frames:
                print(
                    "OMPT-OMPD mismatch: ompt_task_frames (%s) do not match OMPD task frames (%s)!"
                    % (ompt_task_frames, ompd_task_frames)
                )

        # compare task data
        if "ompt_task_data" in field_names:
            ompt_task_data = thread_data["ompt_task_data"].dereference()["value"]
            current_task_handle = curr_thread.get_current_task_handle()
            ompd_value = ompdModule.call_ompd_get_tool_data(6, current_task_handle)[0]
            if ompt_task_data != ompd_value:
                print(
                    "OMPT-OMPD mismatch: value of ompt_task_data (%d) does not match that of OMPD data union (%d)!"
                    % (ompt_task_data, ompd_value)
                )

    def save_thread_object(self, thread_num, thread_id, addr_space):
        """Saves thread object for thread_num inside threads dictionary."""
        thread_handle = ompdModule.get_thread_handle(thread_id, addr_space)
        self.threads[int(thread_num)] = ompd_thread(thread_handle)

    def get_thread(self, thread_num):
        """Get thread object from map."""
        return self.threads[int(thread_num)]

    def get_curr_thread(self):
        """Get current thread object from map or add new one to map, if missing."""
        thread_num = int(gdb.selected_thread().num)
        if thread_num not in self.threads:
            self.add_thread()
        return self.threads[thread_num]

    def add_thread(self):
        """Add currently selected (*) thread to dictionary threads."""
        inf_thread = gdb.selected_thread()
        try:
            self.save_thread_object(inf_thread.num, inf_thread.ptid[1], self.addr_space)
        except:
            traceback.print_exc()

    def list_threads(self, verbose):
        """Prints OpenMP threads only that are being tracking inside the "threads" dictionary.
        See handle_stop_event and add_thread.
        """
        list_tids = []
        curr_inferior = gdb.selected_inferior()

        for inf_thread in curr_inferior.threads():
            list_tids.append((inf_thread.num, inf_thread.ptid))
        if verbose:
            if self.states is None:
                self.enumerate_states()
            for (thread_num, thread_ptid) in sorted(list_tids):
                if thread_num in self.threads:
                    try:
                        print(
                            "Thread %i (%i) is an OpenMP thread; state: %s"
                            % (
                                thread_num,
                                thread_ptid[1],
                                self.states[self.threads[thread_num].get_state()[0]],
                            )
                        )
                    except:
                        traceback.print_exc()
                else:
                    print(
                        "Thread %i (%i) is no OpenMP thread"
                        % (thread_num, thread_ptid[1])
                    )

    def enumerate_states(self):
        """Helper function for list_threads: initializes map of OMPD states for output of
        'ompd threads'.
        """
        if self.states is None:
            self.states = {}
            current = int("0x102", 0)
            count = 0
            more = 1

            while more > 0:
                tup = ompdModule.call_ompd_enumerate_states(self.addr_space, current)
                (next_state, next_state_name, more) = tup

                self.states[next_state] = next_state_name
                current = next_state

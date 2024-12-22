"""
Test the lldb command line completion mechanism.
"""


import os
from multiprocessing import Process
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


class CommandLineCompletionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
        except:
            pass

    def test_at(self):
        """Test that 'at' completes to 'attach '."""
        self.complete_from_to("at", "attach ")

    def test_de(self):
        """Test that 'de' completes to 'detach '."""
        self.complete_from_to("de", "detach ")

    def test_frame_variable(self):
        self.build()

        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertState(process.GetState(), lldb.eStateStopped)

        # Since CommandInterpreter has been corrected to update the current execution
        # context at the beginning of HandleCompletion, we're here explicitly testing
        # the scenario where "frame var" is completed without any preceding commands.
        self.do_test_variable_completion("frame variable")

    def test_dwim_print(self):
        self.build()

        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertState(process.GetState(), lldb.eStateStopped)

        # Since CommandInterpreter has been corrected to update the current execution
        # context at the beginning of HandleCompletion, we're here explicitly testing
        # the scenario where "frame var" is completed without any preceding commands.
        self.do_test_variable_completion("dwim-print")

    def do_test_variable_completion(self, command):
        self.complete_from_to(f"{command} fo", f"{command} fooo")
        self.complete_from_to(f"{command} fooo.", f"{command} fooo.t")
        self.complete_from_to(f"{command} fooo.t.", f"{command} fooo.t.x")
        self.complete_from_to(f"{command} fooo.dd", f"{command} fooo.dd")

        self.complete_from_to(f"{command} ptr_fooo->", f"{command} ptr_fooo->t")
        self.complete_from_to(f"{command} ptr_fooo->t.", f"{command} ptr_fooo->t.x")
        self.complete_from_to(f"{command} ptr_fooo->dd", f"{command} ptr_fooo->dd")

        self.complete_from_to(f"{command} cont", f"{command} container")
        self.complete_from_to(f"{command} container.", f"{command} container.MemberVar")
        self.complete_from_to(
            f"{command} container.Mem", f"{command} container.MemberVar"
        )

        self.complete_from_to(f"{command} ptr_cont", f"{command} ptr_container")
        self.complete_from_to(
            f"{command} ptr_container->", f"{command} ptr_container->MemberVar"
        )
        self.complete_from_to(
            f"{command} ptr_container->Mem", f"{command} ptr_container->MemberVar"
        )

    def test_process_attach_dash_dash_con(self):
        """Test that 'process attach --con' completes to 'process attach --continue '."""
        self.complete_from_to("process attach --con", "process attach --continue ")

    def test_process_launch_arch(self):
        self.complete_from_to("process launch --arch ", ["mips", "arm64"])

    def test_process_load(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.complete_from_to("process load Makef", "process load Makefile")

    @skipUnlessPlatform(["linux"])
    def test_process_unload(self):
        """Test the completion for "process unload <index>" """
        # This tab completion should not work without a running process.
        self.complete_from_to("process unload ", "process unload ")

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        err = lldb.SBError()
        local_spec = lldb.SBFileSpec(self.getBuildArtifact("libshared.so"))
        if lldb.remote_platform:
            self.process().LoadImage(
                local_spec,
                lldb.SBFileSpec(
                    lldbutil.append_to_process_working_directory(self, "libshared.so"),
                    False,
                ),
                err,
            )
        else:
            self.process().LoadImage(local_spec, err)
        self.assertSuccess(err)

        self.complete_from_to("process unload ", "process unload 0")

        self.process().UnloadImage(0)
        self.complete_from_to("process unload ", "process unload ")

    def test_process_plugin_completion(self):
        subcommands = ["attach -P", "connect -p", "launch -p"]

        for subcommand in subcommands:
            self.complete_from_to(
                "process " + subcommand + " mac",
                "process " + subcommand + " mach-o-core",
            )

    def completions_contain_str(self, input, needle):
        interp = self.dbg.GetCommandInterpreter()
        match_strings = lldb.SBStringList()
        num_matches = interp.HandleCompletion(input, len(input), 0, -1, match_strings)
        found_needle = False
        for match in match_strings:
            if needle in match:
                found_needle = True
                break
        self.assertTrue(
            found_needle, "Returned completions: " + "\n".join(match_strings)
        )

    @skipIfRemote
    def test_common_completion_process_pid_and_name(self):
        # The LLDB process itself and the process already attached to are both
        # ignored by the process discovery mechanism, thus we need a process known
        # to us here.
        self.build()
        server = self.spawnSubprocess(
            self.getBuildArtifact("a.out"),
            [
                "-x"
            ],  # Arg "-x" makes the subprocess wait for input thus it won't be terminated too early
            install_remote=False,
        )
        self.assertIsNotNone(server)
        pid = server.pid

        self.completions_contain("process attach -p ", [str(pid)])
        self.completions_contain("platform process attach -p ", [str(pid)])
        self.completions_contain("platform process info ", [str(pid)])

        self.completions_contain_str("process attach -n ", "a.out")
        self.completions_contain_str("platform process attach -n ", "a.out")

    def test_process_signal(self):
        # The tab completion for "process signal"  won't work without a running process.
        self.complete_from_to("process signal ", "process signal ")

        # Test with a running process.
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        lldbutil.run_to_source_breakpoint(self, "// Break here", self.main_source_spec)

        self.complete_from_to("process signal ", "process signal SIG")
        self.complete_from_to("process signal SIGPIP", "process signal SIGPIPE")
        self.complete_from_to("process signal SIGA", ["SIGABRT", "SIGALRM"])

    def test_ambiguous_long_opt(self):
        self.completions_match(
            "breakpoint modify --th", ["--thread-id", "--thread-index", "--thread-name"]
        )

    def test_disassemble_dash_f(self):
        self.completions_match("disassemble -F ", ["default", "intel", "att"])

    def test_plugin_load(self):
        self.complete_from_to("plugin load ", [])

    def test_log_enable(self):
        self.complete_from_to("log disable ll", ["lldb"])
        self.complete_from_to("log disable dw", ["dwarf"])
        self.complete_from_to("log disable lldb al", ["all"])
        self.complete_from_to("log disable lldb sym", ["symbol"])

    def test_log_list(self):
        self.complete_from_to("log list ll", ["lldb"])
        self.complete_from_to("log list dw", ["dwarf"])
        self.complete_from_to("log list ll", ["lldb"])
        self.complete_from_to("log list lldb dwa", ["dwarf"])

    def test_quoted_command(self):
        self.complete_from_to('"set', ['"settings" '])

    def test_quoted_arg_with_quoted_command(self):
        self.complete_from_to('"settings" "repl', ['"replace" '])

    def test_quoted_arg_without_quoted_command(self):
        self.complete_from_to('settings "repl', ['"replace" '])

    def test_single_quote_command(self):
        self.complete_from_to("'set", ["'settings' "])

    def test_terminated_quote_command(self):
        # This should not crash, but we don't get any
        # reasonable completions from this.
        self.complete_from_to("'settings'", [])

    def test_process_launch_arch_arm(self):
        self.complete_from_to("process launch --arch arm", ["arm64"])

    def test_target_symbols_add_shlib(self):
        # Doesn't seem to work, but at least it shouldn't crash.
        self.complete_from_to("target symbols add --shlib ", [])

    def test_log_file(self):
        # Complete in our source directory which contains a 'main.cpp' file.
        src_dir = self.getSourceDir() + "/"
        self.complete_from_to("log enable lldb expr -f " + src_dir, ["main.cpp"])

    def test_log_dir(self):
        # Complete our source directory.
        src_dir = os.path.dirname(os.path.realpath(__file__))
        self.complete_from_to("log enable lldb expr -f " + src_dir, [src_dir + os.sep])

    # <rdar://problem/11052829>
    def test_infinite_loop_while_completing(self):
        """Test that 'process print hello\' completes to itself and does not infinite loop."""
        self.complete_from_to("process print hello\\", "process print hello\\")

    def test_watchpoint_co(self):
        """Test that 'watchpoint co' completes to 'watchpoint command '."""
        self.complete_from_to("watchpoint co", "watchpoint command ")

    def test_watchpoint_command_space(self):
        """Test that 'watchpoint command ' completes to ['add', 'delete', 'list']."""
        self.complete_from_to("watchpoint command ", ["add", "delete", "list"])

    def test_watchpoint_command_a(self):
        """Test that 'watchpoint command a' completes to 'watchpoint command add '."""
        self.complete_from_to("watchpoint command a", "watchpoint command add ")

    def test_watchpoint_set_ex(self):
        """Test that 'watchpoint set ex' completes to 'watchpoint set expression '."""
        self.complete_from_to("watchpoint set ex", "watchpoint set expression ")

    def test_watchpoint_set_var(self):
        """Test that 'watchpoint set var' completes to 'watchpoint set variable '."""
        self.complete_from_to("watchpoint set var", "watchpoint set variable ")

    def test_watchpoint_set_variable_foo(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.complete_from_to(
            "watchpoint set variable fo", "watchpoint set variable fooo"
        )
        # Only complete the first argument.
        self.complete_from_to(
            "watchpoint set variable fooo ", "watchpoint set variable fooo "
        )

    def test_help_fi(self):
        """Test that 'help fi' completes to ['file', 'finish']."""
        self.complete_from_to("help fi", ["file", "finish"])

    def test_help_watchpoint_s(self):
        """Test that 'help watchpoint s' completes to 'help watchpoint set '."""
        self.complete_from_to("help watchpoint s", "help watchpoint set ")

    @expectedFailureNetBSD
    @add_test_categories(["watchpoint"])
    def test_common_complete_watchpoint_ids(self):
        subcommands = ["enable", "disable", "delete", "modify", "ignore"]

        # Completion should not work without a target.
        for subcommand in subcommands:
            self.complete_from_to(
                "watchpoint " + subcommand + " ", "watchpoint " + subcommand + " "
            )

        # Create a process to provide a target and enable watchpoint setting.
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("watchpoint set variable ptr_fooo")
        for subcommand in subcommands:
            self.complete_from_to("watchpoint " + subcommand + " ", ["1"])

    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to(
            "settings append target.er", "settings append target.error-path"
        )

    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.env' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to(
            "settings insert-after target.env", "settings insert-after target.env-vars"
        )

    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.env' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to(
            "settings insert-before target.env",
            "settings insert-before target.env-vars",
        )

    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to(
            "settings replace target.ru", "settings replace target.run-args"
        )

    def test_settings_show_term(self):
        self.complete_from_to("settings show term-", "settings show term-width")

    def test_settings_list_term(self):
        self.complete_from_to("settings list term-", "settings list term-width")

    def test_settings_remove_term(self):
        self.complete_from_to("settings remove term-", "settings remove term-width")

    def test_settings_s(self):
        """Test that 'settings s' completes to ['set', 'show']."""
        self.complete_from_to("settings s", ["set", "show"])

    def test_settings_set_th(self):
        """Test that 'settings set thread-f' completes to 'settings set thread-format'."""
        self.complete_from_to("settings set thread-f", "settings set thread-format")

    def test_settings_s_dash(self):
        """Test that 'settings set --g' completes to 'settings set --global'."""
        self.complete_from_to("settings set --g", "settings set --global")

    def test_settings_clear_th(self):
        """Test that 'settings clear thread-f' completes to 'settings clear thread-format'."""
        self.complete_from_to("settings clear thread-f", "settings clear thread-format")

    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to("settings set target.ma", "settings set target.max-")

    def test_settings_set_target_exec(self):
        """Test that 'settings set target.exec' completes to 'settings set target.exec-search-paths '."""
        self.complete_from_to(
            "settings set target.exec", "settings set target.exec-search-paths"
        )

    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to [
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to(
            "settings set target.pr", ["target.prefer-dynamic-value", "target.process."]
        )

    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to(
            "settings set target.process", "settings set target.process."
        )

    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.t' completes to 'settings set target.process.thread.'."""
        self.complete_from_to(
            "settings set target.process.thr", "settings set target.process.thread."
        )

    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to [
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread'].
        """
        self.complete_from_to(
            "settings set target.process.thread.",
            [
                "target.process.thread.step-avoid-regexp",
                "target.process.thread.trace-thread",
            ],
        )

    def test_settings_set_can_complete_setting_enum_values(self):
        """Checks that we can complete the values of an enum setting."""
        self.complete_from_to(
            "settings set stop-disassembly-display ",
            ["never", "always", "no-debuginfo", "no-source"],
        )

    def test_thread_plan_discard(self):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "ptr_foo", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(thread)
        self.complete_from_to("thread plan discard ", "thread plan discard ")

        source_path = os.path.join(self.getSourceDir(), "thread_plan_script.py")
        self.runCmd("command script import '%s'" % (source_path))
        self.runCmd("thread step-scripted -C thread_plan_script.PushPlanStack")
        self.complete_from_to("thread plan discard ", "thread plan discard 1")
        self.runCmd("thread plan discard 1")

    def test_target_space(self):
        """Test that 'target ' completes to ['create', 'delete', 'list',
        'modules', 'select', 'stop-hook', 'variable']."""
        self.complete_from_to(
            "target ",
            ["create", "delete", "list", "modules", "select", "stop-hook", "variable"],
        )

    def test_target_modules_dump_line_table(self):
        """Tests source file completion by completing the line-table argument."""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to("target modules dump line-table main.cp", ["main.cpp"])

    def test_custom_command_completion(self):
        """Tests completion in custom user provided commands."""
        completion_types = [
            "none",
            "source-file",
            "disk-file",
            "disk-directory",
            "symbol",
            "module",
            "settings-name",
            "platform-plugin",
            "architecture",
            "variable-path",
            "register",
            "breakpoint",
            "process-plugin",
            "disassembly-flavor",
            "type-language",
            "frame-index",
            "module-uuid",
            "stophook-id",
            "thread-index",
            "watchpoint-id",
            "breakpoint-name",
            "process-id",
            "process-name",
            "remote-disk-file",
            "remote-disk-directory",
            "type-category-name",
            "custom",
        ]
        self.completions_contain("command script add -C ", completion_types)

        source_path = os.path.join(self.getSourceDir(), "my_test_cmd.py")
        self.runCmd("command script import '%s'" % (source_path))
        self.runCmd(
            "command script add -C disk-file -f my_test_cmd.my_test_cmd my_test_cmd"
        )
        self.complete_from_to("my_test_cmd main.cp", ["main.cpp"])
        self.expect("my_test_cmd main.cpp", substrs=["main.cpp"])

    @skipIf(hostoslist=["windows"])
    def test_completion_target_create_from_root_dir(self):
        """Tests source file completion by completing ."""
        root_dir = os.path.abspath(os.sep)
        self.completions_contain(
            "target create " + root_dir,
            list(
                filter(
                    lambda x: os.path.exists(x) and os.path.isdir(x),
                    map(lambda x: root_dir + x + os.sep, os.listdir(root_dir)),
                )
            ),
        )

    def test_target_modules_load_aout(self):
        """Tests modules completion by completing the target modules load argument."""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to("target modules load a.ou", ["a.out"])

    def test_target_modules_search_paths_insert(self):
        # Completion won't work without a valid target.
        self.complete_from_to(
            "target modules search-paths insert ", "target modules search-paths insert "
        )
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.complete_from_to(
            "target modules search-paths insert ", "target modules search-paths insert "
        )
        self.runCmd("target modules search-paths add a b")
        self.complete_from_to(
            "target modules search-paths insert ",
            "target modules search-paths insert 0",
        )
        # Completion only works for the first arg.
        self.complete_from_to(
            "target modules search-paths insert 0 ",
            "target modules search-paths insert 0 ",
        )

    def test_target_create_dash_co(self):
        """Test that 'target create --co' completes to 'target variable --core '."""
        self.complete_from_to("target create --co", "target create --core ")

    def test_target_va(self):
        """Test that 'target va' completes to 'target variable '."""
        self.complete_from_to("target va", "target variable ")

    def test_common_completion_thread_index(self):
        subcommands = [
            "continue",
            "info",
            "exception",
            "select",
            "step-in",
            "step-inst",
            "step-inst-over",
            "step-out",
            "step-over",
            "step-script",
        ]

        # Completion should do nothing without threads.
        for subcommand in subcommands:
            self.complete_from_to(
                "thread " + subcommand + " ", "thread " + subcommand + " "
            )

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )

        # At least we have the thread at the index of 1 now.
        for subcommand in subcommands:
            self.complete_from_to("thread " + subcommand + " ", ["1"])

    def test_common_completion_type_category_name(self):
        subcommands = ["delete", "list", "enable", "disable", "define"]
        for subcommand in subcommands:
            self.complete_from_to("type category " + subcommand + " ", ["default"])
        self.complete_from_to("type filter add -w ", ["default"])

    def test_command_argument_completion(self):
        """Test completion of command arguments"""
        self.complete_from_to("watchpoint set variable -", ["-w", "-s"])
        self.complete_from_to(
            "watchpoint set variable -w", "watchpoint set variable -w "
        )
        self.complete_from_to("watchpoint set variable --", ["--watch", "--size"])
        self.complete_from_to(
            "watchpoint set variable --w", "watchpoint set variable --watch"
        )
        self.complete_from_to(
            "watchpoint set variable -w ", ["read", "write", "read_write"]
        )
        self.complete_from_to(
            "watchpoint set variable --watch ", ["read", "write", "read_write"]
        )
        self.complete_from_to(
            "watchpoint set variable --watch w", "watchpoint set variable --watch write"
        )
        self.complete_from_to(
            "watchpoint set variable -w read_", "watchpoint set variable -w read_write"
        )
        # Now try the same thing with a variable name (non-option argument) to
        # test that getopts arg reshuffling doesn't confuse us.
        self.complete_from_to("watchpoint set variable foo -", ["-w", "-s"])
        self.complete_from_to(
            "watchpoint set variable foo -w", "watchpoint set variable foo -w "
        )
        self.complete_from_to("watchpoint set variable foo --", ["--watch", "--size"])
        self.complete_from_to(
            "watchpoint set variable foo --w", "watchpoint set variable foo --watch"
        )
        self.complete_from_to(
            "watchpoint set variable foo -w ", ["read", "write", "read_write"]
        )
        self.complete_from_to(
            "watchpoint set variable foo --watch ", ["read", "write", "read_write"]
        )
        self.complete_from_to(
            "watchpoint set variable foo --watch w",
            "watchpoint set variable foo --watch write",
        )
        self.complete_from_to(
            "watchpoint set variable foo -w read_",
            "watchpoint set variable foo -w read_write",
        )

    def test_command_script_delete(self):
        self.runCmd("command script add -h test_desc -f none -s current usercmd1")
        self.check_completion_with_desc("command script delete ", [["usercmd1", ""]])

    def test_command_delete(self):
        self.runCmd(
            r"command regex test_command s/^$/finish/ 's/([0-9]+)/frame select %1/'"
        )
        self.complete_from_to("command delete test_c", "command delete test_command")

    def test_command_unalias(self):
        self.complete_from_to("command unalias ima", "command unalias image")

    def test_command_aliases(self):
        self.runCmd("command alias brkpt breakpoint")
        # Exact matches are chosen if possible, even if there are longer
        # completions we could use.
        self.complete_from_to("b", "b ")
        # Aliases are included in possible completions.
        self.complete_from_to("br", ["breakpoint", "brkpt"])
        # An alias can be the chosen completion.
        self.complete_from_to("brk", "brkpt")

        # The list can contain only aliases if there's no built-ins to match.
        self.runCmd("command alias test_1 breakpoint")
        self.runCmd("command alias test_2 breakpoint")
        self.complete_from_to("test_", ["test_1", "test_2"])

    def test_completion_description_commands(self):
        """Test descriptions of top-level command completions"""
        self.check_completion_with_desc(
            "",
            [
                ["command", "Commands for managing custom LLDB commands."],
                [
                    "breakpoint",
                    "Commands for operating on breakpoints (see 'help b' for shorthand.)",
                ],
            ],
        )

        self.check_completion_with_desc(
            "pl",
            [
                ["platform", "Commands to manage and create platforms."],
                ["plugin", "Commands for managing LLDB plugins."],
            ],
        )

        # Just check that this doesn't crash.
        self.check_completion_with_desc("comman", [])
        self.check_completion_with_desc("non-existent-command", [])

    def test_completion_description_command_options(self):
        """Test descriptions of command options"""
        # Short options
        self.check_completion_with_desc(
            "breakpoint set -",
            [
                ["-h", "Set the breakpoint on exception catcH."],
                ["-w", "Set the breakpoint on exception throW."],
            ],
        )

        # Long options.
        self.check_completion_with_desc(
            "breakpoint set --",
            [
                ["--on-catch", "Set the breakpoint on exception catcH."],
                ["--on-throw", "Set the breakpoint on exception throW."],
            ],
        )

        # Ambiguous long options.
        self.check_completion_with_desc(
            "breakpoint set --on-",
            [
                ["--on-catch", "Set the breakpoint on exception catcH."],
                ["--on-throw", "Set the breakpoint on exception throW."],
            ],
        )

        # Unknown long option.
        self.check_completion_with_desc("breakpoint set --Z", [])

    def test_common_completion_frame_index(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )

        self.complete_from_to("frame select ", ["0"])
        self.complete_from_to("thread backtrace -s ", ["0"])

    def test_frame_recognizer_delete(self):
        self.runCmd(
            "frame recognizer add -l py_class -s module_name -n recognizer_name"
        )
        self.check_completion_with_desc(
            "frame recognizer delete ",
            [["0", "py_class, module module_name, demangled symbol recognizer_name"]],
        )

    def test_platform_install_local_file(self):
        self.complete_from_to(
            "platform target-install main.cp", "platform target-install main.cpp"
        )

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_symbol_name(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to(
            "breakpoint set -n Fo", "breakpoint set -n Foo::Bar(int,\\ int)"
        )
        # No completion for Qu because the candidate is
        # (anonymous namespace)::Quux().
        self.complete_from_to("breakpoint set -n Qu", "")

    def test_completion_type_formatter_delete(self):
        self.runCmd("type filter add --child a Aoo")
        self.complete_from_to("type filter delete ", ["Aoo"])
        self.runCmd("type filter add --child b -x Boo")
        self.complete_from_to("type filter delete ", ["Boo"])

        self.runCmd("type format add -f hex Coo")
        self.complete_from_to("type format delete ", ["Coo"])
        self.runCmd("type format add -f hex -x Doo")
        self.complete_from_to("type format delete ", ["Doo"])

        self.runCmd("type summary add -c Eoo")
        self.complete_from_to("type summary delete ", ["Eoo"])
        self.runCmd("type summary add -x -c Foo")
        self.complete_from_to("type summary delete ", ["Foo"])

        self.runCmd("type synthetic add Goo -l test")
        self.complete_from_to("type synthetic delete ", ["Goo"])
        self.runCmd("type synthetic add -x Hoo -l test")
        self.complete_from_to("type synthetic delete ", ["Hoo"])

    def test_register_no_complete(self):
        # The tab completion for "register read/write"  won't work without a running process.
        self.complete_from_to("register read ", "register read ")
        self.complete_from_to("register write ", "register write ")
        self.complete_from_to("register info ", "register info ")

        self.build()
        self.runCmd("target create {}".format(self.getBuildArtifact("a.out")))
        self.runCmd("run")

        # Once a program has finished you have an execution context but no register
        # context so completion cannot work.
        self.complete_from_to("register read ", "register read ")
        self.complete_from_to("register write ", "register write ")
        self.complete_from_to("register info ", "register info ")

    @skipIf(archs=no_match(["x86_64"]))
    def test_register_read_and_write_on_x86(self):
        """Test the completion of the commands register read and write on x86"""

        self.build()
        self.main_source_spec = lldb.SBFileSpec("main.cpp")
        lldbutil.run_to_source_breakpoint(self, "// Break here", self.main_source_spec)

        # test cases for register read
        self.complete_from_to("register read ", ["rax", "rbx", "rcx"])
        self.complete_from_to("register read r", ["rax", "rbx", "rcx"])
        self.complete_from_to("register read ra", "register read rax")
        # register read can take multiple register names as arguments
        self.complete_from_to("register read rax ", ["rax", "rbx", "rcx"])
        # complete with prefix '$'
        self.completions_match("register read $rb", ["$rbx", "$rbp"])
        self.completions_match("register read $ra", ["$rax"])
        self.complete_from_to("register read rax $", ["$rax", "$rbx", "$rcx"])
        self.complete_from_to("register read $rax ", ["rax", "rbx", "rcx"])

        # test cases for register write
        self.complete_from_to("register write ", ["rax", "rbx", "rcx"])
        self.complete_from_to("register write r", ["rax", "rbx", "rcx"])
        self.complete_from_to("register write ra", "register write rax")
        self.complete_from_to("register write rb", ["rbx", "rbp"])
        # register write can only take exact one register name as argument
        self.complete_from_to("register write rbx ", [])

    def test_common_completion_target_stophook_ids(self):
        subcommands = ["delete", "enable", "disable"]

        for subcommand in subcommands:
            self.complete_from_to(
                "target stop-hook " + subcommand + " ",
                "target stop-hook " + subcommand + " ",
            )

        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.runCmd("target stop-hook add -o test")

        for subcommand in subcommands:
            self.complete_from_to(
                "target stop-hook " + subcommand + " ",
                "target stop-hook " + subcommand + " 1",
            )

        # Completion should work only on the first argument.
        for subcommand in subcommands:
            self.complete_from_to(
                "target stop-hook " + subcommand + " 1 ",
                "target stop-hook " + subcommand + " 1 ",
            )

    def test_common_completion_type_language(self):
        self.complete_from_to("type category -l ", ["c"])

    def test_target_modules_load_dash_u(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to(
            "target modules load -u ", [target.GetModuleAtIndex(0).GetUUIDString()]
        )

    def test_complete_breakpoint_with_ids(self):
        """These breakpoint subcommands should be completed with a list of breakpoint ids"""

        subcommands = [
            "enable",
            "disable",
            "delete",
            "modify",
            "name add",
            "name delete",
            "write",
        ]

        # The tab completion here is unavailable without a target
        for subcommand in subcommands:
            self.complete_from_to(
                "breakpoint " + subcommand + " ", "breakpoint " + subcommand + " "
            )

        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        bp = target.BreakpointCreateByName("main", "a.out")
        self.assertTrue(bp)
        self.assertEqual(bp.GetNumLocations(), 1)

        for subcommand in subcommands:
            self.complete_from_to("breakpoint " + subcommand + " ", ["1"])

        bp2 = target.BreakpointCreateByName("Bar", "a.out")
        self.assertTrue(bp2)
        self.assertEqual(bp2.GetNumLocations(), 1)

        for subcommand in subcommands:
            self.complete_from_to("breakpoint " + subcommand + " ", ["1", "2"])

        for subcommand in subcommands:
            self.complete_from_to("breakpoint " + subcommand + " 1 ", ["1", "2"])

    def test_complete_breakpoint_with_names(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # test breakpoint read dedicated
        self.complete_from_to("breakpoint read -N ", "breakpoint read -N ")
        self.complete_from_to("breakpoint read -f breakpoints.json -N ", ["mm"])
        self.complete_from_to(
            "breakpoint read -f breakpoints.json -N n",
            "breakpoint read -f breakpoints.json -N n",
        )
        self.complete_from_to(
            "breakpoint read -f breakpoints_invalid.json -N ",
            "breakpoint read -f breakpoints_invalid.json -N ",
        )

        # test common breapoint name completion
        bp1 = target.BreakpointCreateByName("main", "a.out")
        self.assertTrue(bp1)
        self.assertEqual(bp1.GetNumLocations(), 1)
        self.complete_from_to("breakpoint set -N n", "breakpoint set -N n")
        self.assertTrue(bp1.AddNameWithErrorHandling("nn"))
        self.complete_from_to("breakpoint set -N ", "breakpoint set -N nn")

    def test_ambiguous_command(self):
        """Test completing an ambiguous commands"""
        self.complete_from_to("settings s", ["set", "show"])

    def test_ambiguous_subcommand(self):
        """Test completing a subcommand of an ambiguous command"""
        self.complete_from_to("settings s ta", [])

    def test_shlib_name(self):
        self.build()
        error = lldb.SBError()
        # Create a target, but don't load dependent modules
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"), None, None, False, error)
        self.assertSuccess(error)
        self.registerSharedLibrariesWithTarget(target, ["shared"])

        basenames = []
        paths = []
        for m in target.modules:
            basenames.append(m.file.basename)
            paths.append(m.file.fullpath)

        # To see all the diffs
        self.maxDiff = None

        # An empty string completes to everything
        self.completions_match("target symbols add -s ", basenames + paths)

        # Base name completion
        self.completions_match("target symbols add -s a.", ["a.out"])

        # Full path completion
        prefix = os.path.commonpath(paths)
        self.completions_match("target symbols add -s '" + prefix, paths)

        # Full path, but ending with a path separator
        prefix_sep = prefix + os.path.sep
        self.completions_match("target symbols add -s '" + prefix_sep, paths)

        # The completed path should match the spelling of the input, so if the
        # input contains a double separator, so should the completions.
        prefix_sep_sep = prefix_sep + os.path.sep
        paths_sep = [prefix_sep_sep + p[len(prefix_sep) :] for p in paths]
        self.completions_match("target symbols add -s '" + prefix_sep_sep, paths_sep)

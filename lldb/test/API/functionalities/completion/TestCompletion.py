"""
Test the lldb command line completion mechanism.
"""



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil


class CommandLineCompletionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

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
        self.complete_from_to('at', 'attach ')

    def test_de(self):
        """Test that 'de' completes to 'detach '."""
        self.complete_from_to('de', 'detach ')

    def test_frame_variable(self):
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)
        self.assertEquals(process.GetState(), lldb.eStateStopped)
        
        # Since CommandInterpreter has been corrected to update the current execution 
        # context at the beginning of HandleCompletion, we're here explicitly testing  
        # the scenario where "frame var" is completed without any preceding commands.

        self.complete_from_to('frame variable fo',
                              'frame variable fooo')
        self.complete_from_to('frame variable fooo.',
                              'frame variable fooo.')
        self.complete_from_to('frame variable fooo.dd',
                              'frame variable fooo.dd')

        self.complete_from_to('frame variable ptr_fooo->',
                              'frame variable ptr_fooo->')
        self.complete_from_to('frame variable ptr_fooo->dd',
                              'frame variable ptr_fooo->dd')

        self.complete_from_to('frame variable cont',
                              'frame variable container')
        self.complete_from_to('frame variable container.',
                              'frame variable container.MemberVar')
        self.complete_from_to('frame variable container.Mem',
                              'frame variable container.MemberVar')

        self.complete_from_to('frame variable ptr_cont',
                              'frame variable ptr_container')
        self.complete_from_to('frame variable ptr_container->',
                              'frame variable ptr_container->MemberVar')
        self.complete_from_to('frame variable ptr_container->Mem',
                              'frame variable ptr_container->MemberVar')

    def test_process_attach_dash_dash_con(self):
        """Test that 'process attach --con' completes to 'process attach --continue '."""
        self.complete_from_to(
            'process attach --con',
            'process attach --continue ')

    def test_process_launch_arch(self):
        self.complete_from_to('process launch --arch ',
                              ['mips',
                               'arm64'])

    def test_process_plugin_completion(self):
        subcommands = ['attach -P', 'connect -p', 'launch -p']

        for subcommand in subcommands:
            self.complete_from_to('process ' + subcommand + ' mac',
                                  'process ' + subcommand + ' mach-o-core')

    def test_process_signal(self):
        # The tab completion for "process signal"  won't work without a running process.
        self.complete_from_to('process signal ',
                              'process signal ')

        # Test with a running process.
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        lldbutil.run_to_source_breakpoint(self, '// Break here', self.main_source_spec)

        self.complete_from_to('process signal ',
                              'process signal SIG')
        self.complete_from_to('process signal SIGA',
                              ['SIGABRT',
                               'SIGALRM'])

    def test_ambiguous_long_opt(self):
        self.completions_match('breakpoint modify --th',
                               ['--thread-id',
                                '--thread-index',
                                '--thread-name'])

    def test_plugin_load(self):
        self.complete_from_to('plugin load ', [])

    def test_log_enable(self):
        self.complete_from_to('log enable ll', ['lldb'])
        self.complete_from_to('log enable dw', ['dwarf'])
        self.complete_from_to('log enable lldb al', ['all'])
        self.complete_from_to('log enable lldb sym', ['symbol'])

    def test_log_enable(self):
        self.complete_from_to('log disable ll', ['lldb'])
        self.complete_from_to('log disable dw', ['dwarf'])
        self.complete_from_to('log disable lldb al', ['all'])
        self.complete_from_to('log disable lldb sym', ['symbol'])

    def test_log_list(self):
        self.complete_from_to('log list ll', ['lldb'])
        self.complete_from_to('log list dw', ['dwarf'])
        self.complete_from_to('log list ll', ['lldb'])
        self.complete_from_to('log list lldb dwa', ['dwarf'])

    def test_quoted_command(self):
        self.complete_from_to('"set',
                              ['"settings" '])

    def test_quoted_arg_with_quoted_command(self):
        self.complete_from_to('"settings" "repl',
                              ['"replace" '])

    def test_quoted_arg_without_quoted_command(self):
        self.complete_from_to('settings "repl',
                              ['"replace" '])

    def test_single_quote_command(self):
        self.complete_from_to("'set",
                              ["'settings' "])

    def test_terminated_quote_command(self):
        # This should not crash, but we don't get any
        # reasonable completions from this.
        self.complete_from_to("'settings'", [])

    def test_process_launch_arch_arm(self):
        self.complete_from_to('process launch --arch arm',
                              ['arm64'])

    def test_target_symbols_add_shlib(self):
        # Doesn't seem to work, but at least it shouldn't crash.
        self.complete_from_to('target symbols add --shlib ', [])

    def test_log_file(self):
        # Complete in our source directory which contains a 'main.cpp' file.
        src_dir =  os.path.dirname(os.path.realpath(__file__)) + '/'
        self.complete_from_to('log enable lldb expr -f ' + src_dir,
                              ['main.cpp'])

    def test_log_dir(self):
        # Complete our source directory.
        src_dir =  os.path.dirname(os.path.realpath(__file__))
        self.complete_from_to('log enable lldb expr -f ' + src_dir,
                              [src_dir + os.sep], turn_off_re_match=True)

    # <rdar://problem/11052829>
    def test_infinite_loop_while_completing(self):
        """Test that 'process print hello\' completes to itself and does not infinite loop."""
        self.complete_from_to('process print hello\\', 'process print hello\\',
                              turn_off_re_match=True)

    def test_watchpoint_co(self):
        """Test that 'watchpoint co' completes to 'watchpoint command '."""
        self.complete_from_to('watchpoint co', 'watchpoint command ')

    def test_watchpoint_command_space(self):
        """Test that 'watchpoint command ' completes to ['add', 'delete', 'list']."""
        self.complete_from_to(
            'watchpoint command ', [
                'add', 'delete', 'list'])

    def test_watchpoint_command_a(self):
        """Test that 'watchpoint command a' completes to 'watchpoint command add '."""
        self.complete_from_to(
            'watchpoint command a',
            'watchpoint command add ')

    def test_watchpoint_set_ex(self):
        """Test that 'watchpoint set ex' completes to 'watchpoint set expression '."""
        self.complete_from_to(
            'watchpoint set ex',
            'watchpoint set expression ')

    def test_watchpoint_set_var(self):
        """Test that 'watchpoint set var' completes to 'watchpoint set variable '."""
        self.complete_from_to('watchpoint set var', 'watchpoint set variable ')

    def test_help_fi(self):
        """Test that 'help fi' completes to ['file', 'finish']."""
        self.complete_from_to(
            'help fi', [
                'file', 'finish'])

    def test_help_watchpoint_s(self):
        """Test that 'help watchpoint s' completes to 'help watchpoint set '."""
        self.complete_from_to('help watchpoint s', 'help watchpoint set ')

    def test_settings_append_target_er(self):
        """Test that 'settings append target.er' completes to 'settings append target.error-path'."""
        self.complete_from_to(
            'settings append target.er',
            'settings append target.error-path')

    def test_settings_insert_after_target_en(self):
        """Test that 'settings insert-after target.env' completes to 'settings insert-after target.env-vars'."""
        self.complete_from_to(
            'settings insert-after target.env',
            'settings insert-after target.env-vars')

    def test_settings_insert_before_target_en(self):
        """Test that 'settings insert-before target.env' completes to 'settings insert-before target.env-vars'."""
        self.complete_from_to(
            'settings insert-before target.env',
            'settings insert-before target.env-vars')

    def test_settings_replace_target_ru(self):
        """Test that 'settings replace target.ru' completes to 'settings replace target.run-args'."""
        self.complete_from_to(
            'settings replace target.ru',
            'settings replace target.run-args')

    def test_settings_show_term(self):
        self.complete_from_to(
            'settings show term-',
            'settings show term-width')

    def test_settings_list_term(self):
        self.complete_from_to(
            'settings list term-',
            'settings list term-width')

    def test_settings_remove_term(self):
        self.complete_from_to(
            'settings remove term-',
            'settings remove term-width')

    def test_settings_s(self):
        """Test that 'settings s' completes to ['set', 'show']."""
        self.complete_from_to(
            'settings s', [
                'set', 'show'])

    def test_settings_set_th(self):
        """Test that 'settings set thread-f' completes to 'settings set thread-format'."""
        self.complete_from_to('settings set thread-f', 'settings set thread-format')

    def test_settings_s_dash(self):
        """Test that 'settings set --g' completes to 'settings set --global'."""
        self.complete_from_to('settings set --g', 'settings set --global')

    def test_settings_clear_th(self):
        """Test that 'settings clear thread-f' completes to 'settings clear thread-format'."""
        self.complete_from_to(
            'settings clear thread-f',
            'settings clear thread-format')

    def test_settings_set_ta(self):
        """Test that 'settings set ta' completes to 'settings set target.'."""
        self.complete_from_to(
            'settings set target.ma',
            'settings set target.max-')

    def test_settings_set_target_exec(self):
        """Test that 'settings set target.exec' completes to 'settings set target.exec-search-paths '."""
        self.complete_from_to(
            'settings set target.exec',
            'settings set target.exec-search-paths')

    def test_settings_set_target_pr(self):
        """Test that 'settings set target.pr' completes to [
        'target.prefer-dynamic-value', 'target.process.']."""
        self.complete_from_to('settings set target.pr',
                              ['target.prefer-dynamic-value',
                               'target.process.'])

    def test_settings_set_target_process(self):
        """Test that 'settings set target.process' completes to 'settings set target.process.'."""
        self.complete_from_to(
            'settings set target.process',
            'settings set target.process.')

    def test_settings_set_target_process_dot(self):
        """Test that 'settings set target.process.t' completes to 'settings set target.process.thread.'."""
        self.complete_from_to(
            'settings set target.process.t',
            'settings set target.process.thread.')

    def test_settings_set_target_process_thread_dot(self):
        """Test that 'settings set target.process.thread.' completes to [
        'target.process.thread.step-avoid-regexp', 'target.process.thread.trace-thread']."""
        self.complete_from_to('settings set target.process.thread.',
                              ['target.process.thread.step-avoid-regexp',
                               'target.process.thread.trace-thread'])

    def test_target_space(self):
        """Test that 'target ' completes to ['create', 'delete', 'list',
        'modules', 'select', 'stop-hook', 'variable']."""
        self.complete_from_to('target ',
                              ['create',
                               'delete',
                               'list',
                               'modules',
                               'select',
                               'stop-hook',
                               'variable'])

    def test_target_modules_dump_line_table(self):
        """Tests source file completion by completing the line-table argument."""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to('target modules dump line-table main.cp',
                              ['main.cpp'])

    def test_target_modules_load_aout(self):
        """Tests modules completion by completing the target modules load argument."""
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to('target modules load a.ou',
                              ['a.out'])

    def test_target_create_dash_co(self):
        """Test that 'target create --co' completes to 'target variable --core '."""
        self.complete_from_to('target create --co', 'target create --core ')

    def test_target_va(self):
        """Test that 'target va' completes to 'target variable '."""
        self.complete_from_to('target va', 'target variable ')

    def test_command_argument_completion(self):
        """Test completion of command arguments"""
        self.complete_from_to("watchpoint set variable -", ["-w", "-s"])
        self.complete_from_to('watchpoint set variable -w', 'watchpoint set variable -w ')
        self.complete_from_to("watchpoint set variable --", ["--watch", "--size"])
        self.complete_from_to("watchpoint set variable --w", "watchpoint set variable --watch")
        self.complete_from_to('watchpoint set variable -w ', ['read', 'write', 'read_write'])
        self.complete_from_to("watchpoint set variable --watch ", ["read", "write", "read_write"])
        self.complete_from_to("watchpoint set variable --watch w", "watchpoint set variable --watch write")
        self.complete_from_to('watchpoint set variable -w read_', 'watchpoint set variable -w read_write')
        # Now try the same thing with a variable name (non-option argument) to
        # test that getopts arg reshuffling doesn't confuse us.
        self.complete_from_to("watchpoint set variable foo -", ["-w", "-s"])
        self.complete_from_to('watchpoint set variable foo -w', 'watchpoint set variable foo -w ')
        self.complete_from_to("watchpoint set variable foo --", ["--watch", "--size"])
        self.complete_from_to("watchpoint set variable foo --w", "watchpoint set variable foo --watch")
        self.complete_from_to('watchpoint set variable foo -w ', ['read', 'write', 'read_write'])
        self.complete_from_to("watchpoint set variable foo --watch ", ["read", "write", "read_write"])
        self.complete_from_to("watchpoint set variable foo --watch w", "watchpoint set variable foo --watch write")
        self.complete_from_to('watchpoint set variable foo -w read_', 'watchpoint set variable foo -w read_write')

    def test_completion_description_commands(self):
        """Test descriptions of top-level command completions"""
        self.check_completion_with_desc("", [
            ["command", "Commands for managing custom LLDB commands."],
            ["breakpoint", "Commands for operating on breakpoints (see 'help b' for shorthand.)"]
        ])

        self.check_completion_with_desc("pl", [
            ["platform", "Commands to manage and create platforms."],
            ["plugin", "Commands for managing LLDB plugins."]
        ])

        # Just check that this doesn't crash.
        self.check_completion_with_desc("comman", [])
        self.check_completion_with_desc("non-existent-command", [])

    def test_completion_description_command_options(self):
        """Test descriptions of command options"""
        # Short options
        self.check_completion_with_desc("breakpoint set -", [
            ["-h", "Set the breakpoint on exception catcH."],
            ["-w", "Set the breakpoint on exception throW."]
        ])

        # Long options.
        self.check_completion_with_desc("breakpoint set --", [
            ["--on-catch", "Set the breakpoint on exception catcH."],
            ["--on-throw", "Set the breakpoint on exception throW."]
        ])

        # Ambiguous long options.
        self.check_completion_with_desc("breakpoint set --on-", [
            ["--on-catch", "Set the breakpoint on exception catcH."],
            ["--on-throw", "Set the breakpoint on exception throW."]
        ])

        # Unknown long option.
        self.check_completion_with_desc("breakpoint set --Z", [
        ])

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_symbol_name(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.complete_from_to('breakpoint set -n Fo',
                              'breakpoint set -n Foo::Bar(int,\\ int)',
                              turn_off_re_match=True)
        # No completion for Qu because the candidate is
        # (anonymous namespace)::Quux().
        self.complete_from_to('breakpoint set -n Qu', '')

    @skipIf(archs=no_match(['x86_64']))
    def test_register_read_and_write_on_x86(self):
        """Test the completion of the commands register read and write on x86"""

        # The tab completion for "register read/write"  won't work without a running process.
        self.complete_from_to('register read ',
                              'register read ')
        self.complete_from_to('register write ',
                              'register write ')

        self.build()
        self.main_source_spec = lldb.SBFileSpec("main.cpp")
        lldbutil.run_to_source_breakpoint(self, '// Break here', self.main_source_spec)

        # test cases for register read
        self.complete_from_to('register read ',
                              ['rax',
                               'rbx',
                               'rcx'])
        self.complete_from_to('register read r',
                              ['rax',
                               'rbx',
                               'rcx'])
        self.complete_from_to('register read ra',
                              'register read rax')
        # register read can take multiple register names as arguments
        self.complete_from_to('register read rax ',
                              ['rax',
                               'rbx',
                               'rcx'])
        # complete with prefix '$'
        self.completions_match('register read $rb',
                              ['$rbx',
                               '$rbp'])
        self.completions_match('register read $ra',
                              ['$rax'])
        self.complete_from_to('register read rax $',
                              ['\$rax',
                               '\$rbx',
                               '\$rcx'])
        self.complete_from_to('register read $rax ',
                              ['rax',
                               'rbx',
                               'rcx'])

        # test cases for register write
        self.complete_from_to('register write ',
                              ['rax',
                               'rbx',
                               'rcx'])
        self.complete_from_to('register write r',
                              ['rax',
                               'rbx',
                               'rcx'])
        self.complete_from_to('register write ra',
                              'register write rax')
        self.complete_from_to('register write rb',
                              ['rbx',
                               'rbp'])
        # register write can only take exact one register name as argument
        self.complete_from_to('register write rbx ',
                              [])

    def test_complete_breakpoint_with_ids(self):
        """These breakpoint subcommands should be completed with a list of breakpoint ids"""

        subcommands = ['enable', 'disable', 'delete', 'modify', 'name add', 'name delete', 'write']

        # The tab completion here is unavailable without a target
        for subcommand in subcommands:
            self.complete_from_to('breakpoint ' + subcommand + ' ',
                                  'breakpoint ' + subcommand + ' ')

        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact('a.out'))
        self.assertTrue(target, VALID_TARGET)

        bp = target.BreakpointCreateByName('main', 'a.out')
        self.assertTrue(bp)
        self.assertEqual(bp.GetNumLocations(), 1)

        for subcommand in subcommands:
            self.complete_from_to('breakpoint ' + subcommand + ' ',
                                  ['1'])
        
        bp2 = target.BreakpointCreateByName('Bar', 'a.out')
        self.assertTrue(bp2)
        self.assertEqual(bp2.GetNumLocations(), 1)

        for subcommand in subcommands:
            self.complete_from_to('breakpoint ' + subcommand + ' ',
                                  ['1',
                                   '2'])
        
        for subcommand in subcommands:
            self.complete_from_to('breakpoint ' + subcommand + ' 1 ',
                                  ['1',
                                   '2'])


"""
Test the lldb command line completion mechanism for the 'expr' command.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbutil

class CommandLineExprCompletionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_expr_completion(self):
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        # Try the completion before we have a context to complete on.
        self.assume_no_completions('expr some_expr')
        self.assume_no_completions('expr ')
        self.assume_no_completions('expr f')


        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)

        # Completing member functions
        self.complete_exactly('expr some_expr.FooNoArgs',
                              'expr some_expr.FooNoArgsBar()')
        self.complete_exactly('expr some_expr.FooWithArgs',
                              'expr some_expr.FooWithArgsBar(')
        self.complete_exactly('expr some_expr.FooWithMultipleArgs',
                              'expr some_expr.FooWithMultipleArgsBar(')
        self.complete_exactly('expr some_expr.FooUnderscore',
                              'expr some_expr.FooUnderscoreBar_()')
        self.complete_exactly('expr some_expr.FooNumbers',
                              'expr some_expr.FooNumbersBar1()')
        self.complete_exactly('expr some_expr.StaticMemberMethod',
                              'expr some_expr.StaticMemberMethodBar()')

        # Completing static functions
        self.complete_exactly('expr Expr::StaticMemberMethod',
                              'expr Expr::StaticMemberMethodBar()')

        # Completing member variables
        self.complete_exactly('expr some_expr.MemberVariab',
                              'expr some_expr.MemberVariableBar')

        # Multiple completions
        self.completions_contain('expr some_expr.',
                                 ['some_expr.FooNumbersBar1()',
                                  'some_expr.FooUnderscoreBar_()',
                                  'some_expr.FooWithArgsBar(',
                                  'some_expr.MemberVariableBar'])

        self.completions_contain('expr some_expr.Foo',
                                 ['some_expr.FooNumbersBar1()',
                                  'some_expr.FooUnderscoreBar_()',
                                  'some_expr.FooWithArgsBar('])

        self.completions_contain('expr ',
                                 ['static_cast',
                                  'reinterpret_cast',
                                  'dynamic_cast'])

        self.completions_contain('expr 1 + ',
                                 ['static_cast',
                                  'reinterpret_cast',
                                  'dynamic_cast'])

        # Completion expr without spaces
        # This is a bit awkward looking for the user, but that's how
        # the completion API works at the moment.
        self.completions_contain('expr 1+',
                                 ['1+some_expr', "1+static_cast"])

        # Test with spaces
        self.complete_exactly('expr   some_expr .FooNoArgs',
                              'expr   some_expr .FooNoArgsBar()')
        self.complete_exactly('expr  some_expr .FooNoArgs',
                              'expr  some_expr .FooNoArgsBar()')
        self.complete_exactly('expr some_expr .FooNoArgs',
                              'expr some_expr .FooNoArgsBar()')
        self.complete_exactly('expr some_expr. FooNoArgs',
                              'expr some_expr. FooNoArgsBar()')
        self.complete_exactly('expr some_expr . FooNoArgs',
                              'expr some_expr . FooNoArgsBar()')
        self.complete_exactly('expr Expr :: StaticMemberMethod',
                              'expr Expr :: StaticMemberMethodBar()')
        self.complete_exactly('expr Expr ::StaticMemberMethod',
                              'expr Expr ::StaticMemberMethodBar()')
        self.complete_exactly('expr Expr:: StaticMemberMethod',
                              'expr Expr:: StaticMemberMethodBar()')

        # Test that string literals don't break our parsing logic.
        self.complete_exactly('expr const char *cstr = "some_e"; char c = *cst',
                              'expr const char *cstr = "some_e"; char c = *cstr')
        self.complete_exactly('expr const char *cstr = "some_e" ; char c = *cst',
                              'expr const char *cstr = "some_e" ; char c = *cstr')
        # Requesting completions inside an incomplete string doesn't provide any
        # completions.
        self.complete_exactly('expr const char *cstr = "some_e',
                              'expr const char *cstr = "some_e')

        # Completing inside double dash should do nothing
        self.assume_no_completions('expr -i0 -- some_expr.', 10)
        self.assume_no_completions('expr -i0 -- some_expr.', 11)

        # Test with expr arguments
        self.complete_exactly('expr -i0 -- some_expr .FooNoArgs',
                              'expr -i0 -- some_expr .FooNoArgsBar()')
        self.complete_exactly('expr  -i0 -- some_expr .FooNoArgs',
                              'expr  -i0 -- some_expr .FooNoArgsBar()')

        # Addrof and deref
        self.complete_exactly('expr (*(&some_expr)).FooNoArgs',
                              'expr (*(&some_expr)).FooNoArgsBar()')
        self.complete_exactly('expr (*(&some_expr)) .FooNoArgs',
                              'expr (*(&some_expr)) .FooNoArgsBar()')
        self.complete_exactly('expr (* (&some_expr)) .FooNoArgs',
                              'expr (* (&some_expr)) .FooNoArgsBar()')
        self.complete_exactly('expr (* (& some_expr)) .FooNoArgs',
                              'expr (* (& some_expr)) .FooNoArgsBar()')

        # Addrof and deref (part 2)
        self.complete_exactly('expr (&some_expr)->FooNoArgs',
                              'expr (&some_expr)->FooNoArgsBar()')
        self.complete_exactly('expr (&some_expr) ->FooNoArgs',
                              'expr (&some_expr) ->FooNoArgsBar()')
        self.complete_exactly('expr (&some_expr) -> FooNoArgs',
                              'expr (&some_expr) -> FooNoArgsBar()')
        self.complete_exactly('expr (&some_expr)-> FooNoArgs',
                              'expr (&some_expr)-> FooNoArgsBar()')

        # Builtin arg
        self.complete_exactly('expr static_ca',
                              'expr static_cast')

        # From other files
        self.complete_exactly('expr fwd_decl_ptr->Hidden',
                              'expr fwd_decl_ptr->HiddenMember')


        # Types
        self.complete_exactly('expr LongClassNa',
                              'expr LongClassName')
        self.complete_exactly('expr LongNamespaceName::NestedCla',
                              'expr LongNamespaceName::NestedClass')

        # Namespaces
        self.complete_exactly('expr LongNamespaceNa',
                              'expr LongNamespaceName::')

        # Multiple arguments
        self.complete_exactly('expr &some_expr + &some_e',
                              'expr &some_expr + &some_expr')
        self.complete_exactly('expr SomeLongVarNameWithCapitals + SomeLongVarName',
                              'expr SomeLongVarNameWithCapitals + SomeLongVarNameWithCapitals')
        self.complete_exactly('expr SomeIntVar + SomeIntV',
                              'expr SomeIntVar + SomeIntVar')

        # Multiple statements
        self.complete_exactly('expr long LocalVariable = 0; LocalVaria',
                              'expr long LocalVariable = 0; LocalVariable')

        # Custom Decls
        self.complete_exactly('expr auto l = [](int LeftHandSide, int bx){ return LeftHandS',
                              'expr auto l = [](int LeftHandSide, int bx){ return LeftHandSide')
        self.complete_exactly('expr struct LocalStruct { long MemberName; } ; LocalStruct S; S.Mem',
                              'expr struct LocalStruct { long MemberName; } ; LocalStruct S; S.MemberName')

        # Completing function call arguments
        self.complete_exactly('expr some_expr.FooWithArgsBar(some_exp',
                              'expr some_expr.FooWithArgsBar(some_expr')
        self.complete_exactly('expr some_expr.FooWithArgsBar(SomeIntV',
                              'expr some_expr.FooWithArgsBar(SomeIntVar')
        self.complete_exactly('expr some_expr.FooWithMultipleArgsBar(SomeIntVar, SomeIntVa',
                              'expr some_expr.FooWithMultipleArgsBar(SomeIntVar, SomeIntVar')

        # Function return values
        self.complete_exactly('expr some_expr.Self().FooNoArgs',
                              'expr some_expr.Self().FooNoArgsBar()')
        self.complete_exactly('expr some_expr.Self() .FooNoArgs',
                              'expr some_expr.Self() .FooNoArgsBar()')
        self.complete_exactly('expr some_expr.Self(). FooNoArgs',
                              'expr some_expr.Self(). FooNoArgsBar()')

    def test_expr_completion_with_descriptions(self):
        self.build()
        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)

        self.check_completion_with_desc("expr ", [
            # builtin types have no description.
            ["int", ""],
            ["float", ""],
            # VarDecls have their type as description.
            ["some_expr", "Expr &"],
        ], enforce_order = True)
        self.check_completion_with_desc("expr some_expr.", [
            # Functions have their signature as description.
            ["some_expr.~Expr()", "inline ~Expr()"],
            ["some_expr.operator=(", "inline Expr &operator=(const Expr &)"],
            # FieldDecls have their type as description.
            ["some_expr.MemberVariableBar", "int"],
            ["some_expr.StaticMemberMethodBar()", "static int StaticMemberMethodBar()"],
            ["some_expr.Self()", "Expr &Self()"],
            ["some_expr.FooNoArgsBar()", "int FooNoArgsBar()"],
            ["some_expr.FooWithArgsBar(", "int FooWithArgsBar(int)"],
            ["some_expr.FooNumbersBar1()", "int FooNumbersBar1()"],
            ["some_expr.FooUnderscoreBar_()", "int FooUnderscoreBar_()"],
            ["some_expr.FooWithMultipleArgsBar(", "int FooWithMultipleArgsBar(int, int)"],
        ], enforce_order = True)

    def assume_no_completions(self, str_input, cursor_pos = None):
        interp = self.dbg.GetCommandInterpreter()
        match_strings = lldb.SBStringList()
        if cursor_pos is None:
          cursor_pos = len(str_input)
        num_matches = interp.HandleCompletion(str_input, cursor_pos, 0, -1, match_strings)

        available_completions = []
        for m in match_strings:
            available_completions.append(m)

        self.assertEquals(num_matches, 0, "Got matches, but didn't expect any: " + str(available_completions))

    def completions_contain(self, str_input, items):
        interp = self.dbg.GetCommandInterpreter()
        match_strings = lldb.SBStringList()
        num_matches = interp.HandleCompletion(str_input, len(str_input), 0, -1, match_strings)
        common_match = match_strings.GetStringAtIndex(0)

        for item in items:
            found = False
            for m in match_strings:
                if m == item:
                    found = True
            if not found:
                # Transform match_strings to a python list with strings
                available_completions = []
                for m in match_strings:
                     available_completions.append(m)
                self.assertTrue(found, "Couldn't find completion " + item + " in completions " + str(available_completions))

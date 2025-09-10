"""
Main integration for llvm-lit: This defines a lit test format.
Also contains logic to load benchmark modules.
"""
import lit
import lit.TestRunner
import lit.util
import lit.formats
import litsupport.modules
import litsupport.modules.hash
import litsupport.testfile
import litsupport.testplan
import os


# The ResultCode constructor has been changed recently in lit.  An additional parameter has ben added, which
# results in: TypeError: __new__() takes exactly 4 arguments (3 given)
# However, some users rely on the lit version provided by pypi that does not require or have add_result_category.
# See for more details: http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20200511/780899.html
try:
    NOCHANGE = lit.Test.ResultCode("NOCHANGE", "Executable Unchanged", False)
    NOEXE = lit.Test.ResultCode("NOEXE", "Executable Missing", True)
except TypeError:
    NOCHANGE = lit.Test.ResultCode("NOCHANGE", False)
    NOEXE = lit.Test.ResultCode("NOEXE", True)


class TestSuiteTest(lit.formats.ShTest):
    def __init__(self):
        super(TestSuiteTest, self).__init__()

    def execute(self, test, litConfig):
        config = test.config
        if config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Test is unsupported")
        if litConfig.noExecute:
            return lit.Test.Result(lit.Test.PASS)

        # Parse .test file and initialize context
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        lit.util.mkdir_p(os.path.dirname(tmpBase))
        context = litsupport.testplan.TestContext(test, litConfig, tmpDir, tmpBase)
        litsupport.testfile.parse(context, test.getSourcePath())
        plan = litsupport.testplan.TestPlan()

        # Report missing test executables.
        if not os.path.exists(context.executable):
            return lit.Test.Result(
                NOEXE, "Executable '%s' is missing" % context.executable
            )

        # Skip unchanged tests
        if config.previous_results:
            litsupport.modules.hash.compute(context)
            if litsupport.modules.hash.same_as_previous(context):
                result = lit.Test.Result(
                    NOCHANGE, "Executable identical to previous run"
                )
                val = lit.Test.toMetricValue(context.executable_hash)
                result.addMetric("hash", val)
                return result

        # Let test modules modify the test plan.
        for modulename in config.test_modules:
            module = litsupport.modules.modules.get(modulename)
            if module is None:
                raise Exception("Unknown testmodule '%s'" % modulename)
            module.mutatePlan(context, plan)

        # Execute Test plan
        result = litsupport.testplan.executePlanTestResult(context, plan)

        return result

import lit
import lit.formats

CUSTOM_PASS = lit.Test.ResultCode('CUSTOM_PASS', False)
CUSTOM_FAILURE = lit.Test.ResultCode('CUSTOM_FAILURE', True)

lit.main.add_result_category(CUSTOM_PASS, "My Passed")
lit.main.add_result_category(CUSTOM_FAILURE, "My Failed")


class MyFormat(lit.formats.ShTest):
    def execute(self, test, lit_config):
        result = super(MyFormat, self).execute(test, lit_config)
        if result.code.isFailure:
            result.code = CUSTOM_FAILURE
        else:
            result.code = CUSTOM_PASS
        return result

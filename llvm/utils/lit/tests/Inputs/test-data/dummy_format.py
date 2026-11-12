import os
import configparser

import lit.formats
import lit.Test


class DummyFormat(lit.formats.FileBasedTest):
    def execute(self, test, lit_config):
        # In this dummy format, expect that each test file is actually just a
        # .ini format dump of the results to report.

        source_path = test.getSourcePath()

        cfg = configparser.ConfigParser()
        cfg.read(source_path)

        # Create the basic test result.
        result_code = cfg.get("global", "result_code")
        result_output = cfg.get("global", "result_output")
        result = lit.Test.Result(getattr(lit.Test, result_code), result_output)

        # Load additional metrics.
        for key, value_str in cfg.items("results"):
            value = eval(value_str)
            metric = lit.Test.toMetricValue(value)
            if isinstance(value, int):
                assert isinstance(metric, lit.Test.IntMetricValue)
                assert metric.format() == lit.Test.IntMetricValue(value).format()
            elif isinstance(value, float):
                assert isinstance(metric, lit.Test.RealMetricValue)
                assert metric.format() == lit.Test.RealMetricValue(value).format()
            elif isinstance(value, str):
                assert isinstance(metric, lit.Test.JSONMetricValue)
                assert metric.format() == lit.Test.JSONMetricValue(value).format()
            else:
                raise RuntimeError("unsupported result type")
            result.addMetric(key, metric)

        return result

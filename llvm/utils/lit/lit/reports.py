import abc
import base64
import datetime
import itertools
import json
import os
import tempfile

from xml.sax.saxutils import quoteattr as quo

import lit.Test


def by_suite_and_test_path(test):
    # Suite names are not necessarily unique.  Include object identity in sort
    # key to avoid mixing tests of different suites.
    return (test.suite.name, id(test.suite), test.path_in_suite)


class Report:
    def __init__(self, output_file):
        self.output_file = output_file
        # Set by the option parser later.
        self.use_unique_output_file_name = False

    def write_results(self, tests, elapsed):
        if self.use_unique_output_file_name:
            filename, ext = os.path.splitext(os.path.basename(self.output_file))
            fd, _ = tempfile.mkstemp(
                suffix=ext, prefix=f"{filename}.", dir=os.path.dirname(self.output_file)
            )
            report_file = os.fdopen(fd, "w", encoding="utf-8")
        else:
            # Overwrite if the results already exist.
            report_file = open(self.output_file, "w", encoding="utf-8")

        with report_file:
            self._write_results_to_file(tests, elapsed, report_file)

    @abc.abstractmethod
    def _write_results_to_file(self, tests, elapsed, file):
        """Write test results to the file object "file"."""
        pass


class JsonReport(Report):
    def _write_results_to_file(self, tests, elapsed, file):
        unexecuted_codes = {lit.Test.EXCLUDED, lit.Test.SKIPPED}
        tests = [t for t in tests if t.result.code not in unexecuted_codes]
        # Construct the data we will write.
        data = {}
        # Encode the current lit version as a schema version.
        data["__version__"] = lit.__versioninfo__
        data["elapsed"] = elapsed
        # FIXME: Record some information on the lit configuration used?
        # FIXME: Record information from the individual test suites?

        # Encode the tests.
        data["tests"] = tests_data = []
        for test in tests:
            test_data = {
                "name": test.getFullName(),
                "code": test.result.code.name,
                "output": test.result.output,
                "elapsed": test.result.elapsed,
            }

            # Add test metrics, if present.
            if test.result.metrics:
                test_data["metrics"] = metrics_data = {}
                for key, value in test.result.metrics.items():
                    metrics_data[key] = value.todata()

            # Report micro-tests separately, if present
            if test.result.microResults:
                for key, micro_test in test.result.microResults.items():
                    # Expand parent test name with micro test name
                    parent_name = test.getFullName()
                    micro_full_name = parent_name + ":" + key

                    micro_test_data = {
                        "name": micro_full_name,
                        "code": micro_test.code.name,
                        "output": micro_test.output,
                        "elapsed": micro_test.elapsed,
                    }
                    if micro_test.metrics:
                        micro_test_data["metrics"] = micro_metrics_data = {}
                        for key, value in micro_test.metrics.items():
                            micro_metrics_data[key] = value.todata()

                    tests_data.append(micro_test_data)

            tests_data.append(test_data)

        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


_invalid_xml_chars_dict = {
    c: None for c in range(32) if chr(c) not in ("\t", "\n", "\r")
}


def remove_invalid_xml_chars(s):
    # According to the XML 1.0 spec, control characters other than
    # \t,\r, and \n are not permitted anywhere in the document
    # (https://www.w3.org/TR/xml/#charsets) and therefore this function
    # removes them to produce a valid XML document.
    #
    # Note: In XML 1.1 only \0 is illegal (https://www.w3.org/TR/xml11/#charsets)
    # but lit currently produces XML 1.0 output.
    return s.translate(_invalid_xml_chars_dict)


class XunitReport(Report):
    skipped_codes = {lit.Test.EXCLUDED, lit.Test.SKIPPED, lit.Test.UNSUPPORTED}

    def _write_results_to_file(self, tests, elapsed, file):
        tests.sort(key=by_suite_and_test_path)
        tests_by_suite = itertools.groupby(tests, lambda t: t.suite)

        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<testsuites time="{time:.2f}">\n'.format(time=elapsed))
        for suite, test_iter in tests_by_suite:
            self._write_testsuite(file, suite, list(test_iter))
        file.write("</testsuites>\n")

    def _write_testsuite(self, file, suite, tests):
        skipped = 0
        failures = 0
        time = 0.0

        for t in tests:
            if t.result.code in self.skipped_codes:
                skipped += 1
            if t.isFailure():
                failures += 1
            time += t.result.elapsed or 0.0

        name = suite.config.name.replace(".", "-")
        file.write(
            f'<testsuite name={quo(name)} tests="{len(tests)}" failures="{failures}" skipped="{skipped}" time="{time:.2f}">\n'
        )
        for test in tests:
            self._write_test(file, test, name)
        file.write("</testsuite>\n")

    def _write_test(self, file, test, suite_name):
        path = "/".join(test.path_in_suite[:-1]).replace(".", "_")
        class_name = f"{suite_name}.{path or suite_name}"
        name = test.path_in_suite[-1]
        time = test.result.elapsed or 0.0
        file.write(
            f'<testcase classname={quo(class_name)} name={quo(name)} time="{time:.2f}"'
        )

        if test.isFailure():
            file.write(">\n  <failure><![CDATA[")
            # In the unlikely case that the output contains the CDATA
            # terminator we wrap it by creating a new CDATA block.
            output = test.result.output.replace("]]>", "]]]]><![CDATA[>")
            if isinstance(output, bytes):
                output = output.decode("utf-8", "ignore")

            # Failing test  output sometimes contains control characters like
            # \x1b (e.g. if there was some -fcolor-diagnostics output) which are
            # not allowed inside XML files.
            # This causes problems with CI systems: for example, the Jenkins
            # JUnit XML will throw an exception when ecountering those
            # characters and similar problems also occur with GitLab CI.
            output = remove_invalid_xml_chars(output)
            file.write(output)
            file.write("]]></failure>\n</testcase>\n")
        elif test.result.code in self.skipped_codes:
            reason = self._get_skip_reason(test)
            file.write(f">\n  <skipped message={quo(reason)}/>\n</testcase>\n")
        else:
            file.write("/>\n")

    def _get_skip_reason(self, test):
        code = test.result.code
        if code == lit.Test.EXCLUDED:
            return "Test not selected (--filter, --max-tests)"
        if code == lit.Test.SKIPPED:
            return "Skipped"

        assert code == lit.Test.UNSUPPORTED
        features = test.getMissingRequiredFeatures()
        if features:
            return "Missing required feature(s): " + ", ".join(features)
        return "Unsupported configuration"


def gen_resultdb_test_entry(
    test_name, start_time, elapsed_time, test_output, result_code, is_expected
):
    test_data = {
        "testId": test_name,
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat() + "Z",
        "duration": "%.9fs" % elapsed_time,
        "summary_html": '<p><text-artifact artifact-id="artifact-content-in-request"></p>',
        "artifacts": {
            "artifact-content-in-request": {
                "contents": base64.b64encode(test_output.encode("utf-8")).decode(
                    "utf-8"
                ),
            },
        },
        "expected": is_expected,
    }
    if (
        result_code == lit.Test.PASS
        or result_code == lit.Test.XPASS
        or result_code == lit.Test.FLAKYPASS
        or result_code == lit.Test.FIXED
    ):
        test_data["status"] = "PASS"
    elif result_code == lit.Test.FAIL or result_code == lit.Test.XFAIL:
        test_data["status"] = "FAIL"
    elif (
        result_code == lit.Test.UNSUPPORTED
        or result_code == lit.Test.SKIPPED
        or result_code == lit.Test.EXCLUDED
    ):
        test_data["status"] = "SKIP"
    elif result_code == lit.Test.UNRESOLVED or result_code == lit.Test.TIMEOUT:
        test_data["status"] = "ABORT"
    return test_data


class ResultDBReport(Report):
    def _write_results_to_file(self, tests, elapsed, file):
        unexecuted_codes = {lit.Test.EXCLUDED, lit.Test.SKIPPED}
        tests = [t for t in tests if t.result.code not in unexecuted_codes]
        data = {}
        data["__version__"] = lit.__versioninfo__
        data["elapsed"] = elapsed
        # Encode the tests.
        data["tests"] = tests_data = []
        for test in tests:
            tests_data.append(
                gen_resultdb_test_entry(
                    test_name=test.getFullName(),
                    start_time=test.result.start,
                    elapsed_time=test.result.elapsed,
                    test_output=test.result.output,
                    result_code=test.result.code,
                    is_expected=not test.result.code.isFailure,
                )
            )
            if test.result.microResults:
                for key, micro_test in test.result.microResults.items():
                    # Expand parent test name with micro test name
                    parent_name = test.getFullName()
                    micro_full_name = parent_name + ":" + key + "microres"
                    tests_data.append(
                        gen_resultdb_test_entry(
                            test_name=micro_full_name,
                            start_time=micro_test.start
                            if micro_test.start
                            else test.result.start,
                            elapsed_time=micro_test.elapsed
                            if micro_test.elapsed
                            else test.result.elapsed,
                            test_output=micro_test.output,
                            result_code=micro_test.code,
                            is_expected=not micro_test.code.isFailure,
                        )
                    )

        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


class TimeTraceReport(Report):
    skipped_codes = {lit.Test.EXCLUDED, lit.Test.SKIPPED, lit.Test.UNSUPPORTED}

    def _write_results_to_file(self, tests, elapsed, file):
        # Find when first test started so we can make start times relative.
        first_start_time = min([t.result.start for t in tests])
        events = [
            self._get_test_event(x, first_start_time)
            for x in tests
            if x.result.code not in self.skipped_codes
        ]

        json_data = {"traceEvents": events}

        json.dump(json_data, file, indent=2, sort_keys=True)

    def _get_test_event(self, test, first_start_time):
        test_name = test.getFullName()
        elapsed_time = test.result.elapsed or 0.0
        start_time = test.result.start - first_start_time if test.result.start else 0.0
        pid = test.result.pid or 0
        return {
            "pid": pid,
            "tid": 1,
            "ph": "X",
            "ts": int(start_time * 1000000.0),
            "dur": int(elapsed_time * 1000000.0),
            "name": test_name,
        }


def _wtt_attr(text: str) -> str:
    """Prepare text for a WTT XML attribute value.

    Flattens newlines/tabs to spaces (raw newlines are illegal in XML
    attributes), drops invalid XML chars using the shared helper, then
    quotes. quoteattr adds the quotes.
    """
    text = (
        text.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )
    return quo(remove_invalid_xml_chars(text))


class WttReport(Report):
    def write_results(self, tests, elapsed: float) -> None:
        with open(self.output_file, "w", encoding="utf-16") as f:
            self._write_results_to_file(tests, elapsed, f)

    def _write_results_to_file(self, tests, elapsed: float, file) -> None:
        tests.sort(key=by_suite_and_test_path)

        machine = os.getenv("COMPUTERNAME", "")
        pid = os.getpid()

        starts = [t.result.start for t in tests if t.result and t.result.start]
        base = min(starts) if starts else 0.0
        base_dt = datetime.datetime.fromtimestamp(base)
        base_time = base_dt.strftime("%Y:%m:%d %H:%M:%S") + ":%03d" % (
            base_dt.microsecond // 1000
        )

        def times(test) -> tuple[int, int]:
            elapsed_time = test.result.elapsed or 0.0
            start_time = test.result.start - base if test.result.start else 0.0
            return int(start_time), int(start_time + elapsed_time)

        root_ctx = 1
        end_ticks = int(elapsed)

        def rc(ref_ctx: str) -> str:
            return f'\t<rti id="" />\n\t<ctx id="{ref_ctx}" />\n'

        file.write('<?xml version="1.0" encoding="utf-16"?>\n')
        file.write("<WTT-Logger>\n")

        file.write(
            f'<RTI ID="" Machine="{machine}" ProcessName="lit" '
            f'ProcessID="{pid}" ThreadID="0" '
            f'BaseTime="{base_time}" Frequency="1" />\n'
        )
        file.write(f'<CTX ID="{root_ctx}" Current="WTTLOG" Parent="ROOT" />\n')

        passed = 0
        failed = 0
        unsupported = 0
        excluded = 0
        skipped = 0

        for test in tests:
            if test.result is None:
                continue

            code = test.result.code
            name = test.getFullName()
            created_at, logged_at = times(test)

            # UNSUPPORTED: report as Pass (feature not applicable on this device).
            if code == lit.Test.UNSUPPORTED:
                file.write(f'<CTX ID="" Current={_wtt_attr(name)} Parent="WTTLOG" />\n')
                file.write(
                    f'<StartTest Title={_wtt_attr(name)} TUID="" CA="{created_at}" LA="{created_at}">\n{rc("")}</StartTest>\n'
                )
                unsupported_msg = _wtt_attr(
                    "UNSUPPORTED on this device; reported as Pass (not applicable)."
                )
                file.write(
                    f'<Msg UserText={unsupported_msg} CA="{logged_at}" LA="{logged_at}">\n{rc("")}</Msg>\n'
                )
                file.write(
                    f'<EndTest Title={_wtt_attr(name)} TUID="" Result="Pass" Repro="" CA="{logged_at}" LA="{logged_at}">\n{rc("")}</EndTest>\n'
                )
                unsupported += 1
                continue

            # EXCLUDED / SKIPPED: omitted from the log and from pass/fail results.
            if code == lit.Test.EXCLUDED:
                excluded += 1
                continue
            if code == lit.Test.SKIPPED:
                skipped += 1
                continue

            if code in (lit.Test.PASS, lit.Test.XFAIL):
                result = "Pass"
                passed += 1
            else:
                result = "Fail"
                failed += 1

            file.write(f'<CTX ID="" Current={_wtt_attr(name)} Parent="WTTLOG" />\n')
            file.write(
                f'<StartTest Title={_wtt_attr(name)} TUID="" CA="{created_at}" LA="{created_at}">\n{rc("")}</StartTest>\n'
            )

            # Write error output for failures (WTT uses <Error>, not <Err>).
            if result == "Fail" and test.result.output:
                error_text = _wtt_attr(test.result.output[:4096])
                file.write(
                    f'<Error UserText={error_text} CA="{logged_at}" LA="{logged_at}">\n{rc("")}</Error>\n'
                )

            if result == "Pass" and test.result.output:
                pass_text = _wtt_attr(test.result.output[:1024])
                file.write(
                    f'<Msg UserText={pass_text} CA="{logged_at}" LA="{logged_at}">\n{rc("")}</Msg>\n'
                )

            file.write(
                f'<EndTest Title={_wtt_attr(name)} TUID="" Result="{result}" Repro="" CA="{logged_at}" LA="{logged_at}">\n{rc("")}</EndTest>\n'
            )

        # Tally of UNSUPPORTED tests reported as Pass.
        if unsupported > 0:
            tally = _wtt_attr(
                f"{unsupported} test(s) were UNSUPPORTED on this device and "
                f"reported as Pass (not applicable)."
            )
            file.write(
                f'<Msg UserText={tally} CA="{end_ticks}" LA="{end_ticks}">\n{rc(root_ctx)}</Msg>\n'
            )

        # Tally of tests omitted from results.
        not_run = excluded + skipped
        if not_run > 0:
            parts = []
            if excluded:
                parts.append(f"{excluded} excluded")
            if skipped:
                parts.append(f"{skipped} skipped")
            tally = _wtt_attr(
                f"{not_run} test(s) were not run ({', '.join(parts)}) and are "
                f"omitted from the pass/fail results."
            )
            file.write(
                f'<Msg UserText={tally} CA="{end_ticks}" LA="{end_ticks}">\n{rc(root_ctx)}</Msg>\n'
            )

        total = passed + failed + unsupported
        file.write(
            f'<PFRollup Total="{total}" Passed="{passed + unsupported}" Failed="{failed}" '
            f'Blocked="0" Warned="0" Skipped="0" CA="{end_ticks}" LA="{end_ticks}">\n'
            f"{rc(root_ctx)}</PFRollup>\n"
        )
        file.write("</WTT-Logger>\n")

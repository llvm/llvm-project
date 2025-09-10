"""Test module to collect code size metrics of the benchmark executable."""
from litsupport import testplan
import logging
import os.path


def _getCodeSize(context):
    # First get the filesize: This should always work.
    metrics = {}
    metrics["size"] = os.path.getsize(context.executable)

    # If we have the llvm-size tool available get the size per segment.
    filename = context.test.getSourcePath()
    if filename.endswith(".test"):
        filename = filename[: -len(".test")]
    filename += ".size"
    if os.path.exists(filename):
        with open(filename, "r") as fp:
            lines = fp.readlines()
        # First line contains executable name, second line should be a
        # "section   size    addr" header, numbers start after that.
        if "section" not in lines[1] or "size" not in lines[1]:
            logging.warning(
                "Unexpected output from llvm-size on '%s'", context.executable
            )
        else:
            for line in lines[2:]:
                line = line.strip()
                if line == "":
                    continue
                values = line.split()
                if len(values) < 2:
                    logging.info("Ignoring malformed output line: %s", line)
                    continue
                if values[0] == "Total":
                    continue
                try:
                    name = values[0]
                    val = int(values[1])
                    metrics["size.%s" % name] = val
                    # The text size output here comes from llvm-size.
                    # Darwin and GNU produce differently-formatted output.
                    # Check that we have exactly one of the valid outputs.
                    assert not (
                        "size.__text" in metrics and "size..text" in metrics
                    ), """Both 'size.__text'
                                and 'size..text' present in metrics.
                                Only one of them should exist."""
                except ValueError:
                    logging.info("Ignoring malformed output line: %s", line)

    return metrics


def mutatePlan(context, plan):
    plan.metric_collectors.append(_getCodeSize)

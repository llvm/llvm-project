"""Test module to collect google benchmark results."""
from litsupport import shellcommand
from litsupport import testplan
import json
import lit.Test


def _mutateCommandLine(context, commandline):
    cmd = shellcommand.parse(commandline)
    cmd.arguments.append("--benchmark_format=json")
    # We need stdout outself to get the benchmark csv data.
    if cmd.stdout is not None:
        raise Exception("Rerouting stdout not allowed for microbenchmarks")
    benchfile = context.tmpBase + ".bench.json"
    cmd.stdout = benchfile
    context.microbenchfiles.append(benchfile)

    return cmd.toCommandline()


def _mutateScript(context, script):
    return testplan.mutateScript(context, script, _mutateCommandLine)


def _collectMicrobenchmarkTime(context, microbenchfiles):
    for f in microbenchfiles:
        content = context.read_result_file(context, f)
        data = json.loads(content)

        # Create a micro_result for each benchmark
        for benchmark in data["benchmarks"]:
            # Name for MicroBenchmark
            name = benchmark["name"]

            # Create Result object with PASS
            microBenchmark = lit.Test.Result(lit.Test.PASS)

            # Add the exec_time metric for this result
            exec_time_metric = lit.Test.toMetricValue(benchmark["cpu_time"])
            microBenchmark.addMetric("exec_time", exec_time_metric)

            # Add Micro Result
            context.micro_results[name] = microBenchmark

    # returning the number of microbenchmarks collected as a metric for the
    # base test
    return {"MicroBenchmarks": lit.Test.toMetricValue(len(context.micro_results))}


def mutatePlan(context, plan):
    context.microbenchfiles = []
    plan.runscript = _mutateScript(context, plan.runscript)
    plan.metric_collectors.append(
        lambda context: _collectMicrobenchmarkTime(context, context.microbenchfiles)
    )

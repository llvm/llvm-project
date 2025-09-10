"""Test module to just run the benchmark. Without this module the benchmark is
not executed. This may be interesting when just collecting compile time and
code size."""


def mutatePlan(context, plan):
    plan.preparescript = context.parsed_preparescript
    plan.runscript = context.parsed_runscript
    plan.verifyscript = context.parsed_verifyscript
    plan.metricscripts = context.parsed_metricscripts

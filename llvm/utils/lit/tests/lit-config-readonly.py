# RUN: not %{lit} %{inputs}/lit-config-readonly 2> %t.err
# RUN: FileCheck < %t.err %s

# CHECK: AttributeError: lit_config.maxIndividualTestTime is read-only. Use config.maxIndividualTestTime instead.

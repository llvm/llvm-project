# RUN: not %{lit} %{inputs}/lit-config-readonly > %t.out 2> %t.err
# RUN: FileCheck --check-prefix=CHECK-ERR < %t.err %s

# CHECK-ERR: AttributeError: lit_config.maxIndividualTestTime is read-only. Use config.maxIndividualTestTime instead.

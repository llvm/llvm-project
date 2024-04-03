# REQUIRES: exegesis-can-measure-latency, exegesis-can-measure-uops, x86_64-linux

# Check that when specifying validation counters, the validation counter is
# collected and the information is displayed in the output. Test across
# multiple configurations that need to be wired up separately for validation
# counter support.

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr --validation-counter=instructions-retired | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr --validation-counter=instructions-retired -execution-mode=subprocess | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=uops -opcode-name=ADD64rr --validation-counter=instructions-retired -execution-mode=subprocess | FileCheck %s

# CHECK: instructions-retired: {{[0-9]+}}

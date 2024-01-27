# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Check that we can use the middle-half repetition mode without crashing

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -repetition-mode=middle-half-duplicate
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -repetition-mode=middle-half-loop

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency -opcode-index=-1 --max-configs-per-opcode=1 --benchmark-phase=prepare-snippet --benchmarks-file=-
# FIXME: it would be good to check how many snippets we end up producing,
# but the number is unstable, so for now just check that we do not crash.

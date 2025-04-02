# Only run this on Linux. Running on Windows can take an exorbinant amount of
# time (upwards of ten minutes), and the only place where this functionality is
# really useful is Linux.
# REQUIRES: x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops -opcode-index=-1 --max-configs-per-opcode=1048576 --benchmark-phase=prepare-snippet --benchmarks-file=-
# FIXME: it would be good to check how many snippets we end up producing,
# but the number is unstable, so for now just check that we do not crash.

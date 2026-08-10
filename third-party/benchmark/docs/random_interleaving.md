<a name="interleaving" />

# Random Interleaving

[Random Interleaving](https://github.com/google/benchmark/issues/1051) is a
technique to lower run-to-run variance. It randomly interleaves repetitions of a
microbenchmark with repetitions from other microbenchmarks in the same benchmark
test. Data shows it is able to lower run-to-run variance by
[40%](https://github.com/google/benchmark/issues/1051) on average.

To use, you mainly need to set `--benchmark_enable_random_interleaving=true`,
and optionally specify non-zero repetition count `--benchmark_repetitions=9`
and optionally decrease the per-repetition time `--benchmark_min_time=0.1`.

# RUN: rm -fr %t.json
# RUN: %{lit} --filter 'nonexistent' --allow-empty-runs --time-trace-output %t.json %{inputs}/time-trace
# RUN: not test -e %t.json

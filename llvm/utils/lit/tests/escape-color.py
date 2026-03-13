# cut off the first 9 lines to avoid absolute file paths in the output
# then keep only the next 10 lines to avoid test timing in the output
# RUN: %{lit} %{inputs}/escape-color/color.txt -a | tail -n +10 | head -n 10 > %t
# RUN: diff --strip-trailing-cr %{inputs}/escape-color/color-escaped.txt %t

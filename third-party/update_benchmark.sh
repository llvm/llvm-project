#!/bin/bash

echo "This script deletes `benchmark`, clones it from github, together"
echo "with its dependencies. It then removes .git* files and dirs."
echo "NOTE!!!"
echo "Please double-check the benchmark github wiki for any changes"
echo "to dependencies. Currently, these are limited to googletest."
echo
read -p "Press a key to continue, or Ctrl+C to cancel"

rm -rf benchmark
git clone https://github.com/google/benchmark.git
rm -rf benchmark/.git*
find benchmark/ -name BUILD -delete
find benchmark/ -name BUILD.bazel -delete
find benchmark/ -name WORKSPACE -delete


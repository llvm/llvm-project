#!/bin/sh
# Deprecated, use 'llvm-lit'.

echo "warning: '$0' is deprecated, use 'llvm-lit' instead."
exec llvm-lit "$@"

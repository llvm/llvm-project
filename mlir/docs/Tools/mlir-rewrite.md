# mlir-rewrite

Tool to simplify rewriting .mlir files. There are a couple of build in rewrites
discussed below along with usage.

Note: This is still in very early stage. Its so early its less a tool than a
growing collection of useful functions: to use its best to do what's needed on
a brance by just hacking it (dialects registered, rewrites etc) to say help
ease a rename, upstream useful utility functions, point to ease others
migrating, and then bin eventually. Once there are actually useful parts it
should be refactored same as mlir-opt.

[TOC]

## simple-rename

Rename per op given a substring to a target. The match and replace uses LLVM's
regex sub for the match and replace while the op-name is matched via regular
string comparison. E.g.,

```
mlir-rewrite input.mlir -o output.mlir --simple-rename \
   --simple-rename-op-name="test.concat" --simple-rename-match="axis" \
                                         --simple-rename-replace="bxis"
```

to replace `axis` substring in the text of the range corresponding to
`test.concat` ops with `bxis`.


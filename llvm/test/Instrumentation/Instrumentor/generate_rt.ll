; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-file=%S/rt_config.json -S
; RUN: diff -b rt.c %S/default_rt

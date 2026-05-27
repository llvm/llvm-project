; RUN: rm -rf %t && mkdir -p %t && cd %t
; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/rt_config.json -S
; RUN: diff -b rt.c %S/default_rt

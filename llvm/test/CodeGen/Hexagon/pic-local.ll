; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -relocation-model=pic < %s | FileCheck %s

define private void @f1() {
  ret void
}

define internal void @f2() {
  ret void
}

define ptr @get_f1() {
  ; CHECK:  r0 = add(pc,##.Lf1@PCREL)
  ret ptr @f1
}

define ptr @get_f2() {
  ; CHECK: r0 = add(pc,##f2@PCREL)
  ret ptr @f2
}

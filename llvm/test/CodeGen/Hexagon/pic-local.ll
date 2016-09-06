; RUN: llc -march=hexagon -mcpu=hexagonv5 -relocation-model=pic < %s | FileCheck %s

define private void @f1() {
  ret void
}

define internal void @f2() {
  ret void
}

define void()* @get_f1() {
  ; CHECK:  r0 = add(pc, ##.Lf1@PCREL)
  ret void()* @f1
}

define void()* @get_f2() {
  ; CHECK: r0 = add(pc, ##f2@PCREL)
  ret void()* @f2
}

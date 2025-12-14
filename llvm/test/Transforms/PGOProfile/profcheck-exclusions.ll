; RUN: opt -passes=prof-inject %s -S -o - | FileCheck %s
; RUN: opt -passes=prof-verify %s --disable-output


define void @bar(i1 %c) #0 {
  ret void
}

attributes #0 = { naked }
; CHECK-NOT: !prof

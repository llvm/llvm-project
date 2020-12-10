; RUN: opt < %s -pgo-icall-prom -S -icp-total-percent-threshold=10 | FileCheck %s
; RUN: opt < %s -passes=pgo-icall-prom -S -icp-total-percent-threshold=10 | FileCheck %s

; PR42413: Previously the call promotion code did not correctly update the byval
; attribute. Check that it does. This situation can come up in LTO scenarios
; where struct types end up not matching.

target triple = "i686-unknown-linux-gnu"

%struct.Foo.1 = type { i32 }
%struct.Foo.2 = type { i32 }

@foo = common global i32 (%struct.Foo.2*)* null, align 8

define i32 @func4(%struct.Foo.1* byval(%struct.Foo.1) %p) {
entry:
  %gep = getelementptr inbounds %struct.Foo.1, %struct.Foo.1* %p, i32 0, i32 0
  %v = load i32, i32* %gep
  ret i32 %v
}

define i32 @func5(%struct.Foo.1* byval(%struct.Foo.1) %p) {
entry:
  %gep = getelementptr inbounds %struct.Foo.1, %struct.Foo.1* %p, i32 0, i32 0
  %v = load i32, i32* %gep
  ret i32 %v
}

define i32 @bar(%struct.Foo.2* %f2) {
entry:
  %tmp = load i32 (%struct.Foo.2*)*, i32 (%struct.Foo.2*)** @foo, align 8
  %call = call i32 %tmp(%struct.Foo.2* byval(%struct.Foo.2) %f2), !prof !1
  ret i32 %call
}

!1 = !{!"VP", i32 0, i64 3000, i64 7651369219802541373, i64 1000, i64 3667884930908592509, i64 1000}


; CHECK: define i32 @bar(%struct.Foo.2* %f2)
;     Cast %struct.Foo.2* to %struct.Foo.1* and use byval(%struct.Foo.2).
; CHECK: %[[cast:[^ ]*]] = bitcast %struct.Foo.2* %f2 to %struct.Foo.1*
; CHECK: call i32 @func4(%struct.Foo.1* byval(%struct.Foo.1) %[[cast]])
;     Same but when callee doesn't have explicit byval type.
; CHECK: %[[cast:[^ ]*]] = bitcast %struct.Foo.2* %f2 to %struct.Foo.1*
; CHECK: call i32 @func5(%struct.Foo.1* byval(%struct.Foo.1) %[[cast]])
;     Original call stays the same.
; CHECK: call i32 %tmp(%struct.Foo.2* byval(%struct.Foo.2) %f2)

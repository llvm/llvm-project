; REQUIRES: asserts
; RUN: llc -mtriple=powerpc64le-unknown-unknown -debug-only=legalize-types \
; RUN:   < %s -o /dev/null 2>&1 | FileCheck %s

define i64 @testAddeReturnType(i64 %X, i64 %Z) {
; CHECK: Legally typed node: {{.*}}: i64,glue = adde {{.*}} 
  %cmp = icmp ne i64 %Z, 0
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

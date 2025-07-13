; RUN: llc -mtriple=hexagon -hexagon-eif=0 -ifcvt-limit=0 -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; Check that we constant extended instructions only when necessary.

define i32 @cext_test1(ptr %a) nounwind {
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+##8000)
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},##300000)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}+##4092)
; CHECK-NOT: r{{[0-9]+}} = add(r{{[0-9]+}},##300)
entry:
  %0 = load i32, ptr %a, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i32 2000
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %1, 300000
  br label %return

if.end:
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i32 1023
  %2 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %2, 300
  br label %return

return:
  %retval.0 = phi i32 [ %add, %if.then ], [ %add3, %if.end ]
  ret i32 %retval.0
}

define i32 @cext_test2(ptr %a) nounwind {
; CHECK-NOT: r{{[0-9]+}} = memub(r{{[0-9]+}}+##1023)
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},##300000)
; CHECK: r{{[0-9]+}} = memub(r{{[0-9]+}}+##1024)
; CHECK-NOT: r{{[0-9]+}} = add(r{{[0-9]+}},##6000)
entry:
  %tobool = icmp ne ptr %a, null
  br i1 %tobool, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr inbounds i8, ptr %a, i32 1023
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 300000
  br label %return

if.end:
  %arrayidx1 = getelementptr inbounds i8, ptr %a, i32 1024
  %1 = load i8, ptr %arrayidx1, align 1
  %conv2 = zext i8 %1 to i32
  %add3 = add nsw i32 %conv2, 6000
  br label %return

return:
  %retval.0 = phi i32 [ %add, %if.then ], [ %add3, %if.end ]
  ret i32 %retval.0
}

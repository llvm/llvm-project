; RUN: opt -instcombine -S -o - %s | FileCheck %s

; CHECK-NOT: getelementptr
; CHECK-NOT: ptrtoint
; CHECK: bitcast i8*
%struct.S = type { i32 (...)** }

@_ZL1p = internal constant { i64, i64 } { i64 1, i64 0 }, align 8

define void @_Z1g1S(%struct.S* %s) nounwind {
entry:
  %tmp = load { i64, i64 }* @_ZL1p, align 8
  %memptr.adj = extractvalue { i64, i64 } %tmp, 1
  %0 = bitcast %struct.S* %s to i8*
  %1 = getelementptr inbounds i8* %0, i64 %memptr.adj
  %this.adjusted = bitcast i8* %1 to %struct.S*
  %memptr.ptr = extractvalue { i64, i64 } %tmp, 0
  %2 = and i64 %memptr.ptr, 1
  %memptr.isvirtual = icmp ne i64 %2, 0
  br i1 %memptr.isvirtual, label %memptr.virtual, label %memptr.nonvirtual

memptr.virtual:                                   ; preds = %entry
  %3 = bitcast %struct.S* %this.adjusted to i8**
  %memptr.vtable = load i8** %3
  %4 = sub i64 %memptr.ptr, 1
  %5 = getelementptr i8* %memptr.vtable, i64 %4
  %6 = bitcast i8* %5 to void (%struct.S*)**
  %memptr.virtualfn = load void (%struct.S*)** %6
  br label %memptr.end

memptr.nonvirtual:                                ; preds = %entry
  %memptr.nonvirtualfn = inttoptr i64 %memptr.ptr to void (%struct.S*)*
  br label %memptr.end

memptr.end:                                       ; preds = %memptr.nonvirtual, %memptr.virtual
  %7 = phi void (%struct.S*)* [ %memptr.virtualfn, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  call void %7(%struct.S* %this.adjusted)
  ret void
}

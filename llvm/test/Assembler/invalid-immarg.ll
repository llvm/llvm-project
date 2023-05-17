; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.byval(ptr byval(i32) immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.inalloca(ptr inalloca(i32) immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.inreg(i32 inreg immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.nest(ptr nest immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.sret(ptr sret(i32) immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.zeroext(i32 zeroext immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.signext(i32 signext immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.returned(i32 returned immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.noalias(ptr noalias immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.readnone(ptr readnone immarg)

; CHECK: Attribute 'immarg' is incompatible with other attributes
declare void @llvm.immarg.readonly(ptr readonly immarg)

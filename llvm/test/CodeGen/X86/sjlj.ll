; RUN: llc < %s -mtriple=i386-pc-linux -mcpu=corei7 -relocation-model=static | FileCheck --check-prefix=X86 %s
; RUN: llc < %s -mtriple=i386-pc-linux -mcpu=corei7 -relocation-model=pic | FileCheck --check-prefix=PIC86 %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7 -relocation-model=static | FileCheck --check-prefix=X64 %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7 -relocation-model=pic | FileCheck --check-prefix=PIC64 %s

@buf = internal global [5 x i8*] zeroinitializer

declare i8* @llvm.frameaddress(i32) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare i32 @llvm.eh.sjlj.setjmp(i8*) nounwind

declare void @llvm.eh.sjlj.longjmp(i8*) nounwind

define i32 @sj0() nounwind {
  %fp = tail call i8* @llvm.frameaddress(i32 0)
  store i8* %fp, i8** getelementptr inbounds ([5 x i8*]* @buf, i64 0, i64 0), align 16
  %sp = tail call i8* @llvm.stacksave()
  store i8* %sp, i8** getelementptr inbounds ([5 x i8*]* @buf, i64 0, i64 2), align 16
  %r = tail call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([5 x i8*]* @buf to i8*))
  ret i32 %r
; X86: sj0
; x86: movl %ebp, buf
; X86: movl %esp, buf+8
; x86: movl ${{.*LBB.*}}, buf+4
; X86: ret
; PIC86: sj0
; PIC86: movl %ebp, buf@GOTOFF(%[[GOT:.*]])
; PIC86: movl %esp, buf@GOTOFF+8(%[[GOT]])
; PIC86: leal {{.*LBB.*}}@GOTOFF(%[[GOT]]), %[[LREG:.*]]
; PIC86: movl %[[LREG]], buf@GOTOFF+4
; PIC86: ret
; X64: sj0
; x64: movq %rbp, buf(%rip)
; x64: movq ${{.*LBB.*}}, buf+8(%rip)
; X64: movq %rsp, buf+16(%rip)
; X64: ret
; PIC64: sj0
; PIC64: movq %rbp, buf(%rip)
; PIC64: movq %rsp, buf+16(%rip)
; PIC64: leaq {{.*LBB.*}}(%rip), %[[LREG:.*]]
; PIC64: movq %[[LREG]], buf+8(%rip)
; PIC64: ret
}

define void @lj0() nounwind {
  tail call void @llvm.eh.sjlj.longjmp(i8* bitcast ([5 x i8*]* @buf to i8*))
  unreachable
; X86: lj0
; X86: movl buf, %ebp
; X86: movl buf+4, %[[REG32:.*]]
; X86: movl buf+8, %esp
; X86: jmpl *%[[REG32]]
; X64: lj0
; X64: movq buf(%rip), %rbp
; X64: movq buf+8(%rip), %[[REG64:.*]]
; X64: movq buf+16(%rip), %rsp
; X64: jmpq *%[[REG64]]
}

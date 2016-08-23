; RUN: llc -march=hexagon -hexagon-expand-condsets=0 < %s | FileCheck %s
;
; Expand-condsets eliminates the "mux" instruction, which is what this
; testcase is checking.

%struct._Dnk_filet.1 = type { i16, i8, i32, i8*, i8*, i8*, i8*, i8*, i8*, i32*, [2 x i32], i8*, i8*, i8*, %struct._Mbstatet.0, i8*, [8 x i8], i8 }
%struct._Mbstatet.0 = type { i32, i16, i16 }

@_Stdout = external global %struct._Dnk_filet.1
@.str = external unnamed_addr constant [23 x i8], align 8

; Test that we don't generate a new value compare if the operands are
; the same register.

; CHECK-NOT: cmp.eq([[REG0:(r[0-9]+)]].new, [[REG0]])
; CHECK: cmp.eq([[REG1:(r[0-9]+)]], [[REG1]])

; Function Attrs: nounwind
declare void @fprintf(%struct._Dnk_filet.1* nocapture, i8* nocapture readonly, ...) #1

define void @main() #0 {
entry:
  %0 = load i32*, i32** undef, align 4
  %1 = load i32, i32* undef, align 4
  br i1 undef, label %if.end, label %_ZNSt6vectorIbSaIbEE3endEv.exit

_ZNSt6vectorIbSaIbEE3endEv.exit:
  %2 = icmp slt i32 %1, 0
  %sub5.i.i.i = lshr i32 %1, 5
  %add619.i.i.i = add i32 %sub5.i.i.i, -134217728
  %sub5.i.pn.i.i = select i1 %2, i32 %add619.i.i.i, i32 %sub5.i.i.i
  %storemerge2.i.i = getelementptr inbounds i32, i32* %0, i32 %sub5.i.pn.i.i
  %cmp.i.i = icmp ult i32* %storemerge2.i.i, %0
  %.mux = select i1 %cmp.i.i, i32 0, i32 1
  br i1 undef, label %_ZNSt6vectorIbSaIbEE3endEv.exit57, label %if.end

_ZNSt6vectorIbSaIbEE3endEv.exit57:
  %3 = icmp slt i32 %1, 0
  %sub5.i.i.i44 = lshr i32 %1, 5
  %add619.i.i.i45 = add i32 %sub5.i.i.i44, -134217728
  %sub5.i.pn.i.i46 = select i1 %3, i32 %add619.i.i.i45, i32 %sub5.i.i.i44
  %storemerge2.i.i47 = getelementptr inbounds i32, i32* %0, i32 %sub5.i.pn.i.i46
  %cmp.i38 = icmp ult i32* %storemerge2.i.i47, %0
  %.reg2mem.sroa.0.sroa.0.0.load14.i.reload = select i1 %cmp.i38, i32 0, i32 1
  %cmp = icmp eq i32 %.mux, %.reg2mem.sroa.0.sroa.0.0.load14.i.reload
  br i1 %cmp, label %if.end, label %if.then

if.then:
  call void (%struct._Dnk_filet.1*, i8*, ...) @fprintf(%struct._Dnk_filet.1* @_Stdout, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0), i32 %.mux, i32 %.reg2mem.sroa.0.sroa.0.0.load14.i.reload) #1
  unreachable

if.end:
  br i1 undef, label %_ZNSt6vectorIbSaIbEED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:
  unreachable

_ZNSt6vectorIbSaIbEED2Ev.exit:
  ret void
}

attributes #0 = { "target-cpu"="hexagonv5" }
attributes #1 = { nounwind "target-cpu"="hexagonv5" }

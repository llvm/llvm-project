; RUN: opt < %s -instcombine
; PR9579

define <2 x i16> @entry(<2 x i16> %a) nounwind {
entry:
  %a.addr = alloca <2 x i16>, align 4
  %.compoundliteral = alloca <2 x i16>, align 4
  store <2 x i16> %a, <2 x i16>* %a.addr, align 4
  %tmp = load <2 x i16>* %a.addr, align 4
  store <2 x i16> zeroinitializer, <2 x i16>* %.compoundliteral
  %tmp1 = load <2 x i16>* %.compoundliteral
  %cmp = icmp uge <2 x i16> %tmp, %tmp1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %sext
}

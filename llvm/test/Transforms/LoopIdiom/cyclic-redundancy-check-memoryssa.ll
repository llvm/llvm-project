; RUN: opt -passes='loop-mssa(loop-idiom),print<memoryssa>' -disable-output %s 2>&1 | FileCheck %s

define i16 @crc16.le.tc8(i8 %msg, i16 %checksum) {
; CHECK-LABEL: MemorySSA for function: crc16.le.tc8
; CHECK:         %tbl.ptradd = getelementptr inbounds i16, ptr @.crctable, i64 %indexer.ext
; CHECK-NEXT:  ; MemoryUse({{.*}})
; CHECK-NEXT:    %tbl.ld = load i16, ptr %tbl.ptradd, align 2
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %crc = phi i16 [ %checksum, %entry ], [ %crc.next, %loop ]
  %data = phi i8 [ %msg, %entry ], [ %data.next, %loop ]
  %crc.cast = trunc i16 %crc to i8
  %xor.crc.data = xor i8 %crc.cast, %data
  %and.crc.data = and i8 %xor.crc.data, 1
  %data.next = lshr i8 %data, 1
  %check.sb = icmp eq i8 %and.crc.data, 0
  %crc.shift = lshr i16 %crc, 1
  %crc.xor = xor i16 %crc.shift, -24575
  %crc.next = select i1 %check.sb, i16 %crc.shift, i16 %crc.xor
  %iv.next = add nuw nsw i32 %iv, 1
  %exit.cond = icmp samesign ult i32 %iv, 7
  br i1 %exit.cond, label %loop, label %exit

exit:                                             ; preds = %loop
  ret i16 %crc.next
}

; RUN: llc < %s -march=bpfel -show-mc-encoding | FileCheck %s

%struct.bpf_map_def = type { i32, i32, i32, i32 }
%struct.sk_buff = type opaque

@hash_map = global %struct.bpf_map_def { i32 1, i32 4, i32 8, i32 1024 }, section "maps", align 4

; Function Attrs: nounwind uwtable
define i32 @bpf_prog2(ptr %skb) #0 section "socket2" {
  %key = alloca i32, align 4
  %val = alloca i64, align 8
  %1 = bitcast ptr %skb to ptr
  %2 = call i64 @llvm.bpf.load.half(ptr %1, i64 12) #2
  %3 = icmp eq i64 %2, 34984
  br i1 %3, label %4, label %6

; <label>:4                                       ; preds = %0
  %5 = call i64 @llvm.bpf.load.half(ptr %1, i64 16) #2
  br label %6

; <label>:6                                       ; preds = %4, %0
  %proto.0.i = phi i64 [ %5, %4 ], [ %2, %0 ]
  %nhoff.0.i = phi i64 [ 18, %4 ], [ 14, %0 ]
  %7 = icmp eq i64 %proto.0.i, 33024
  br i1 %7, label %8, label %12

; <label>:8                                       ; preds = %6
  %9 = add i64 %nhoff.0.i, 2
  %10 = call i64 @llvm.bpf.load.half(ptr %1, i64 %9) #2
  %11 = add i64 %nhoff.0.i, 4
  br label %12

; <label>:12                                      ; preds = %8, %6
  %proto.1.i = phi i64 [ %10, %8 ], [ %proto.0.i, %6 ]
  %nhoff.1.i = phi i64 [ %11, %8 ], [ %nhoff.0.i, %6 ]
  switch i64 %proto.1.i, label %flow_dissector.exit.thread [
    i64 2048, label %13
    i64 34525, label %39
  ]

; <label>:13                                      ; preds = %12
  %14 = add i64 %nhoff.1.i, 6
  %15 = call i64 @llvm.bpf.load.half(ptr %1, i64 %14) #2
  %16 = and i64 %15, 16383
  %17 = icmp eq i64 %16, 0
  br i1 %17, label %18, label %.thread.i.i

; <label>:18                                      ; preds = %13
  %19 = add i64 %nhoff.1.i, 9
  %20 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %19) #2
  %21 = icmp eq i64 %20, 47
  br i1 %21, label %28, label %.thread.i.i

.thread.i.i:                                      ; preds = %18, %13
  %22 = phi i64 [ %20, %18 ], [ 0, %13 ]
  %23 = add i64 %nhoff.1.i, 12
  %24 = call i64 @llvm.bpf.load.word(ptr %1, i64 %23) #2
  %25 = add i64 %nhoff.1.i, 16
  %26 = call i64 @llvm.bpf.load.word(ptr %1, i64 %25) #2
  %27 = trunc i64 %26 to i32
  br label %28

; <label>:28                                      ; preds = %.thread.i.i, %18
  %29 = phi i32 [ %27, %.thread.i.i ], [ undef, %18 ]
  %30 = phi i64 [ %22, %.thread.i.i ], [ 47, %18 ]
  %31 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %nhoff.1.i) #2
  %32 = icmp eq i64 %31, 69
  br i1 %32, label %33, label %35

; <label>:33                                      ; preds = %28
  %34 = add i64 %nhoff.1.i, 20
  br label %parse_ip.exit.i

; <label>:35                                      ; preds = %28
  %36 = shl i64 %31, 2
  %37 = and i64 %36, 60
  %38 = add i64 %37, %nhoff.1.i
  br label %parse_ip.exit.i

; <label>:39                                      ; preds = %12
  %40 = add i64 %nhoff.1.i, 6
  %41 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %40) #2
  %42 = add i64 %nhoff.1.i, 8
  %43 = call i64 @llvm.bpf.load.word(ptr %1, i64 %42) #2
  %44 = add i64 %nhoff.1.i, 12
  %45 = call i64 @llvm.bpf.load.word(ptr %1, i64 %44) #2
  %46 = add i64 %nhoff.1.i, 16
  %47 = call i64 @llvm.bpf.load.word(ptr %1, i64 %46) #2
  %48 = add i64 %nhoff.1.i, 20
  %49 = call i64 @llvm.bpf.load.word(ptr %1, i64 %48) #2
  %50 = add i64 %nhoff.1.i, 24
  %51 = call i64 @llvm.bpf.load.word(ptr %1, i64 %50) #2
  %52 = add i64 %nhoff.1.i, 28
  %53 = call i64 @llvm.bpf.load.word(ptr %1, i64 %52) #2
  %54 = add i64 %nhoff.1.i, 32
  %55 = call i64 @llvm.bpf.load.word(ptr %1, i64 %54) #2
  %56 = add i64 %nhoff.1.i, 36
  %57 = call i64 @llvm.bpf.load.word(ptr %1, i64 %56) #2
  %58 = xor i64 %53, %51
  %59 = xor i64 %58, %55
  %60 = xor i64 %59, %57
  %61 = trunc i64 %60 to i32
  %62 = add i64 %nhoff.1.i, 40
  br label %parse_ip.exit.i

parse_ip.exit.i:                                  ; preds = %39, %35, %33
  %63 = phi i32 [ %61, %39 ], [ %29, %33 ], [ %29, %35 ]
  %64 = phi i64 [ %41, %39 ], [ %30, %33 ], [ %30, %35 ]
  %nhoff.2.i = phi i64 [ %62, %39 ], [ %34, %33 ], [ %38, %35 ]
  switch i64 %64, label %187 [
    i64 47, label %65
    i64 4, label %137
    i64 41, label %163
  ]

; <label>:65                                      ; preds = %parse_ip.exit.i
  %66 = call i64 @llvm.bpf.load.half(ptr %1, i64 %nhoff.2.i) #2
  %67 = add i64 %nhoff.2.i, 2
  %68 = call i64 @llvm.bpf.load.half(ptr %1, i64 %67) #2
  %69 = and i64 %66, 1856
  %70 = icmp eq i64 %69, 0
  br i1 %70, label %71, label %187

; <label>:71                                      ; preds = %65
  %72 = lshr i64 %66, 5
  %73 = and i64 %72, 4
  %74 = add i64 %nhoff.2.i, 4
  %..i = add i64 %74, %73
  %75 = and i64 %66, 32
  %76 = icmp eq i64 %75, 0
  %77 = add i64 %..i, 4
  %nhoff.4.i = select i1 %76, i64 %..i, i64 %77
  %78 = and i64 %66, 16
  %79 = icmp eq i64 %78, 0
  %80 = add i64 %nhoff.4.i, 4
  %nhoff.4..i = select i1 %79, i64 %nhoff.4.i, i64 %80
  %81 = icmp eq i64 %68, 33024
  br i1 %81, label %82, label %86

; <label>:82                                      ; preds = %71
  %83 = add i64 %nhoff.4..i, 2
  %84 = call i64 @llvm.bpf.load.half(ptr %1, i64 %83) #2
  %85 = add i64 %nhoff.4..i, 4
  br label %86

; <label>:86                                      ; preds = %82, %71
  %proto.2.i = phi i64 [ %84, %82 ], [ %68, %71 ]
  %nhoff.6.i = phi i64 [ %85, %82 ], [ %nhoff.4..i, %71 ]
  switch i64 %proto.2.i, label %flow_dissector.exit.thread [
    i64 2048, label %87
    i64 34525, label %113
  ]

; <label>:87                                      ; preds = %86
  %88 = add i64 %nhoff.6.i, 6
  %89 = call i64 @llvm.bpf.load.half(ptr %1, i64 %88) #2
  %90 = and i64 %89, 16383
  %91 = icmp eq i64 %90, 0
  br i1 %91, label %92, label %.thread.i4.i

; <label>:92                                      ; preds = %87
  %93 = add i64 %nhoff.6.i, 9
  %94 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %93) #2
  %95 = icmp eq i64 %94, 47
  br i1 %95, label %102, label %.thread.i4.i

.thread.i4.i:                                     ; preds = %92, %87
  %96 = phi i64 [ %94, %92 ], [ 0, %87 ]
  %97 = add i64 %nhoff.6.i, 12
  %98 = call i64 @llvm.bpf.load.word(ptr %1, i64 %97) #2
  %99 = add i64 %nhoff.6.i, 16
  %100 = call i64 @llvm.bpf.load.word(ptr %1, i64 %99) #2
  %101 = trunc i64 %100 to i32
  br label %102

; <label>:102                                     ; preds = %.thread.i4.i, %92
  %103 = phi i32 [ %101, %.thread.i4.i ], [ %63, %92 ]
  %104 = phi i64 [ %96, %.thread.i4.i ], [ 47, %92 ]
  %105 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %nhoff.6.i) #2
  %106 = icmp eq i64 %105, 69
  br i1 %106, label %107, label %109

; <label>:107                                     ; preds = %102
  %108 = add i64 %nhoff.6.i, 20
  br label %187

; <label>:109                                     ; preds = %102
  %110 = shl i64 %105, 2
  %111 = and i64 %110, 60
  %112 = add i64 %111, %nhoff.6.i
  br label %187

; <label>:113                                     ; preds = %86
  %114 = add i64 %nhoff.6.i, 6
  %115 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %114) #2
  %116 = add i64 %nhoff.6.i, 8
  %117 = call i64 @llvm.bpf.load.word(ptr %1, i64 %116) #2
  %118 = add i64 %nhoff.6.i, 12
  %119 = call i64 @llvm.bpf.load.word(ptr %1, i64 %118) #2
  %120 = add i64 %nhoff.6.i, 16
  %121 = call i64 @llvm.bpf.load.word(ptr %1, i64 %120) #2
  %122 = add i64 %nhoff.6.i, 20
  %123 = call i64 @llvm.bpf.load.word(ptr %1, i64 %122) #2
  %124 = add i64 %nhoff.6.i, 24
  %125 = call i64 @llvm.bpf.load.word(ptr %1, i64 %124) #2
  %126 = add i64 %nhoff.6.i, 28
  %127 = call i64 @llvm.bpf.load.word(ptr %1, i64 %126) #2
  %128 = add i64 %nhoff.6.i, 32
  %129 = call i64 @llvm.bpf.load.word(ptr %1, i64 %128) #2
  %130 = add i64 %nhoff.6.i, 36
  %131 = call i64 @llvm.bpf.load.word(ptr %1, i64 %130) #2
  %132 = xor i64 %127, %125
  %133 = xor i64 %132, %129
  %134 = xor i64 %133, %131
  %135 = trunc i64 %134 to i32
  %136 = add i64 %nhoff.6.i, 40
  br label %187

; <label>:137                                     ; preds = %parse_ip.exit.i
  %138 = add i64 %nhoff.2.i, 6
  %139 = call i64 @llvm.bpf.load.half(ptr %1, i64 %138) #2
  %140 = and i64 %139, 16383
  %141 = icmp eq i64 %140, 0
  br i1 %141, label %142, label %.thread.i1.i

; <label>:142                                     ; preds = %137
  %143 = add i64 %nhoff.2.i, 9
  %144 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %143) #2
  %145 = icmp eq i64 %144, 47
  br i1 %145, label %152, label %.thread.i1.i

.thread.i1.i:                                     ; preds = %142, %137
  %146 = phi i64 [ %144, %142 ], [ 0, %137 ]
  %147 = add i64 %nhoff.2.i, 12
  %148 = call i64 @llvm.bpf.load.word(ptr %1, i64 %147) #2
  %149 = add i64 %nhoff.2.i, 16
  %150 = call i64 @llvm.bpf.load.word(ptr %1, i64 %149) #2
  %151 = trunc i64 %150 to i32
  br label %152

; <label>:152                                     ; preds = %.thread.i1.i, %142
  %153 = phi i32 [ %151, %.thread.i1.i ], [ %63, %142 ]
  %154 = phi i64 [ %146, %.thread.i1.i ], [ 47, %142 ]
  %155 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %nhoff.2.i) #2
  %156 = icmp eq i64 %155, 69
  br i1 %156, label %157, label %159

; <label>:157                                     ; preds = %152
  %158 = add i64 %nhoff.2.i, 20
  br label %187

; <label>:159                                     ; preds = %152
  %160 = shl i64 %155, 2
  %161 = and i64 %160, 60
  %162 = add i64 %161, %nhoff.2.i
  br label %187

; <label>:163                                     ; preds = %parse_ip.exit.i
  %164 = add i64 %nhoff.2.i, 6
  %165 = call i64 @llvm.bpf.load.byte(ptr %1, i64 %164) #2
  %166 = add i64 %nhoff.2.i, 8
  %167 = call i64 @llvm.bpf.load.word(ptr %1, i64 %166) #2
  %168 = add i64 %nhoff.2.i, 12
  %169 = call i64 @llvm.bpf.load.word(ptr %1, i64 %168) #2
  %170 = add i64 %nhoff.2.i, 16
  %171 = call i64 @llvm.bpf.load.word(ptr %1, i64 %170) #2
  %172 = add i64 %nhoff.2.i, 20
  %173 = call i64 @llvm.bpf.load.word(ptr %1, i64 %172) #2
  %174 = add i64 %nhoff.2.i, 24
  %175 = call i64 @llvm.bpf.load.word(ptr %1, i64 %174) #2
  %176 = add i64 %nhoff.2.i, 28
  %177 = call i64 @llvm.bpf.load.word(ptr %1, i64 %176) #2
  %178 = add i64 %nhoff.2.i, 32
  %179 = call i64 @llvm.bpf.load.word(ptr %1, i64 %178) #2
  %180 = add i64 %nhoff.2.i, 36
  %181 = call i64 @llvm.bpf.load.word(ptr %1, i64 %180) #2
  %182 = xor i64 %177, %175
  %183 = xor i64 %182, %179
  %184 = xor i64 %183, %181
  %185 = trunc i64 %184 to i32
  %186 = add i64 %nhoff.2.i, 40
  br label %187

; <label>:187                                     ; preds = %163, %159, %157, %113, %109, %107, %65, %parse_ip.exit.i
  %188 = phi i32 [ %63, %parse_ip.exit.i ], [ %185, %163 ], [ %63, %65 ], [ %135, %113 ], [ %103, %107 ], [ %103, %109 ], [ %153, %157 ], [ %153, %159 ]
  %189 = phi i64 [ %64, %parse_ip.exit.i ], [ %165, %163 ], [ 47, %65 ], [ %115, %113 ], [ %104, %107 ], [ %104, %109 ], [ %154, %157 ], [ %154, %159 ]
  %nhoff.7.i = phi i64 [ %nhoff.2.i, %parse_ip.exit.i ], [ %186, %163 ], [ %nhoff.2.i, %65 ], [ %136, %113 ], [ %108, %107 ], [ %112, %109 ], [ %158, %157 ], [ %162, %159 ]
  %cond.i.i = icmp eq i64 %189, 51
  %190 = select i1 %cond.i.i, i64 4, i64 0
  %191 = add i64 %190, %nhoff.7.i
  %192 = call i64 @llvm.bpf.load.word(ptr %1, i64 %191) #2
  store i32 %188, ptr %key, align 4
  %193 = bitcast ptr %key to ptr
  %194 = call ptr inttoptr (i64 1 to ptr)(ptr @hash_map, ptr %193) #2
  %195 = icmp eq ptr %194, null
  br i1 %195, label %199, label %196

; <label>:196                                     ; preds = %187
  %197 = bitcast ptr %194 to ptr
  %198 = atomicrmw add ptr %197, i64 1 seq_cst
  br label %flow_dissector.exit.thread

; <label>:199                                     ; preds = %187
  store i64 1, ptr %val, align 8
  %200 = bitcast ptr %val to ptr
  %201 = call i32 inttoptr (i64 2 to ptr)(ptr @hash_map, ptr %193, ptr %200, i64 0) #2
  br label %flow_dissector.exit.thread

flow_dissector.exit.thread:                       ; preds = %86, %12, %196, %199
  ret i32 0
; CHECK-LABEL: bpf_prog2:
; CHECK: r0 = *(u16 *)skb[12] # encoding: [0x28,0x00,0x00,0x00,0x0c,0x00,0x00,0x00]
; CHECK: r0 = *(u16 *)skb[16] # encoding: [0x28,0x00,0x00,0x00,0x10,0x00,0x00,0x00]
; CHECK: implicit-def: $r8
; CHECK: r1 =
; CHECK: call 1 # encoding: [0x85,0x00,0x00,0x00,0x01,0x00,0x00,0x00]
; CHECK: call 2 # encoding: [0x85,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
}

declare i64 @llvm.bpf.load.half(ptr, i64) #1

declare i64 @llvm.bpf.load.word(ptr, i64) #1

declare i64 @llvm.bpf.load.byte(ptr, i64) #1

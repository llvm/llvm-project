; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX 
 
 ;SKX-LABEL: zext_8x8mem_to_8x16:                  
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1     
;SKX-NEXT:  vpmovzxbw (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq                            
define <8 x i16> @zext_8x8mem_to_8x16(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = zext <8 x i8> %a to <8 x i16>  
  %ret = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> zeroinitializer  
  ret <8 x i16> %ret
}

;SKX-LABEL: sext_8x8mem_to_8x16:                  
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1     
;SKX-NEXT:  vpmovsxbw (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq                       
define <8 x i16> @sext_8x8mem_to_8x16(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = sext <8 x i8> %a to <8 x i16>  
  %ret = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> zeroinitializer  
  ret <8 x i16> %ret
}

;SKX-LABEL: zext_16x8mem_to_16x16:                
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm0, %k1     
;SKX-NEXT:  vpmovzxbw (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq            
define <16 x i16> @zext_16x8mem_to_16x16(<16 x i8> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i8>,<16 x i8> *%i,align 1
  %x   = zext <16 x i8> %a to <16 x i16>  
  %ret = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer  
  ret <16 x i16> %ret
}

;SKX-LABEL: sext_16x8mem_to_16x16:                
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm0, %k1     
;SKX-NEXT:  vpmovsxbw (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq  
define <16 x i16> @sext_16x8mem_to_16x16(<16 x i8> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i8>,<16 x i8> *%i,align 1
  %x   = sext <16 x i8> %a to <16 x i16>  
  %ret = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer  
  ret <16 x i16> %ret
}

;SKX-LABEL: zext_16x8_to_16x16:                   
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovzxbw %xmm0, %ymm0    
;SKX-NEXT:  retq  
define <16 x i16> @zext_16x8_to_16x16(<16 x i8> %a ) nounwind readnone {  
  %x   = zext <16 x i8> %a to <16 x i16>  
  ret <16 x i16> %x
}

;SKX-LABEL: zext_16x8_to_16x16_mask:              
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm1, %k1     
;SKX-NEXT:  vpmovzxbw %xmm0, %ymm0 {%k1} {z} 
;SKX-NEXT:  retq 
define <16 x i16> @zext_16x8_to_16x16_mask(<16 x i8> %a ,<16 x i1> %mask) nounwind readnone {  
  %x   = zext <16 x i8> %a to <16 x i16> 
  %ret = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer   
  ret <16 x i16> %ret
}

;SKX-LABEL: sext_16x8_to_16x16:                   
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxbw %xmm0, %ymm0    
;SKX-NEXT:  retq
define <16 x i16> @sext_16x8_to_16x16(<16 x i8> %a ) nounwind readnone {  
  %x   = sext <16 x i8> %a to <16 x i16>  
  ret <16 x i16> %x
}

;SKX-LABEL: sext_16x8_to_16x16_mask:              
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm1, %k1     
;SKX-NEXT:  vpmovsxbw %xmm0, %ymm0 {%k1} {z} 
;SKX-NEXT:  retq 
define <16 x i16> @sext_16x8_to_16x16_mask(<16 x i8> %a ,<16 x i1> %mask) nounwind readnone {  
  %x   = sext <16 x i8> %a to <16 x i16> 
  %ret = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer   
  ret <16 x i16> %ret
}

;SKX-LABEL: zext_32x8mem_to_32x16:                
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %ymm0, %k1     
;SKX-NEXT:  vpmovzxbw (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq                       
define <32 x i16> @zext_32x8mem_to_32x16(<32 x i8> *%i , <32 x i1> %mask) nounwind readnone {
  %a   = load <32 x i8>,<32 x i8> *%i,align 1
  %x   = zext <32 x i8> %a to <32 x i16>  
  %ret = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer  
  ret <32 x i16> %ret
}

;SKX-LABEL: sext_32x8mem_to_32x16:                
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %ymm0, %k1     
;SKX-NEXT:  vpmovsxbw (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <32 x i16> @sext_32x8mem_to_32x16(<32 x i8> *%i , <32 x i1> %mask) nounwind readnone {
  %a   = load <32 x i8>,<32 x i8> *%i,align 1
  %x   = sext <32 x i8> %a to <32 x i16>  
  %ret = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer  
  ret <32 x i16> %ret
}

;SKX-LABEL: zext_32x8_to_32x16:                   
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovzxbw %ymm0, %zmm0    
;SKX-NEXT:  retq 
define <32 x i16> @zext_32x8_to_32x16(<32 x i8> %a ) nounwind readnone {  
  %x   = zext <32 x i8> %a to <32 x i16>  
  ret <32 x i16> %x
}

;SKX-LABEL: zext_32x8_to_32x16_mask:              
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %ymm1, %k1    
;SKX-NEXT:  vpmovzxbw %ymm0, %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <32 x i16> @zext_32x8_to_32x16_mask(<32 x i8> %a ,<32 x i1> %mask) nounwind readnone {  
  %x   = zext <32 x i8> %a to <32 x i16>
  %ret = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer  
  ret <32 x i16> %ret
}

;SKX-LABEL: sext_32x8_to_32x16:                   
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxbw %ymm0, %zmm0    
;SKX-NEXT:  retq
define <32 x i16> @sext_32x8_to_32x16(<32 x i8> %a ) nounwind readnone {  
  %x   = sext <32 x i8> %a to <32 x i16>  
  ret <32 x i16> %x
}

;SKX-LABEL: sext_32x8_to_32x16_mask:              
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %ymm1, %k1     
;SKX-NEXT:  vpmovsxbw %ymm0, %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <32 x i16> @sext_32x8_to_32x16_mask(<32 x i8> %a ,<32 x i1> %mask) nounwind readnone {  
  %x   = sext <32 x i8> %a to <32 x i16>
  %ret = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer  
  ret <32 x i16> %ret
}

;SKX-LABEL: zext_4x8mem_to_4x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m    %xmm0, %k1      
;SKX-NEXT:  vpmovzxbd    (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq                            
define <4 x i32> @zext_4x8mem_to_4x32(<4 x i8> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i8>,<4 x i8> *%i,align 1
  %x   = zext <4 x i8> %a to <4 x i32>  
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer  
  ret <4 x i32> %ret
}

;SKX-LABEL: sext_4x8mem_to_4x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m    %xmm0, %k1      
;SKX-NEXT:  vpmovsxbd    (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq       
define <4 x i32> @sext_4x8mem_to_4x32(<4 x i8> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i8>,<4 x i8> *%i,align 1
  %x   = sext <4 x i8> %a to <4 x i32>  
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer  
  ret <4 x i32> %ret
}

;SKX-LABEL: zext_8x8mem_to_8x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m    %xmm0, %k1      
;SKX-NEXT:  vpmovzxbd    (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq    
define <8 x i32> @zext_8x8mem_to_8x32(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = zext <8 x i8> %a to <8 x i32>  
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer  
  ret <8 x i32> %ret
}

;SKX-LABEL: sext_8x8mem_to_8x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m    %xmm0, %k1      
;SKX-NEXT:  vpmovsxbd    (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq         
define <8 x i32> @sext_8x8mem_to_8x32(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = sext <8 x i8> %a to <8 x i32>  
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer  
  ret <8 x i32> %ret
}

;KNL-LABEL: zext_16x8mem_to_16x32:   
;KNL:       vpmovzxbd    (%rdi), %zmm0 {%k1} {z} 
;KNL-NEXT:  retq 
define <16 x i32> @zext_16x8mem_to_16x32(<16 x i8> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i8>,<16 x i8> *%i,align 1
  %x   = zext <16 x i8> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;KNL-LABEL: sext_16x8mem_to_16x32:   
;KNL:       vpmovsxbd    (%rdi), %zmm0 {%k1} {z} 
;KNL-NEXT:  retq  
define <16 x i32> @sext_16x8mem_to_16x32(<16 x i8> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i8>,<16 x i8> *%i,align 1
  %x   = sext <16 x i8> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;KNL-LABEL: zext_16x8_to_16x32_mask:                    
;KNL:       vpmovzxbd %xmm0, %zmm0 {%k1} {z} 
;KNL-NEXT:  retq                
define <16 x i32> @zext_16x8_to_16x32_mask(<16 x i8> %a , <16 x i1> %mask) nounwind readnone {
  %x   = zext <16 x i8> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;KNL-LABEL: sext_16x8_to_16x32_mask:                    
;KNL:       vpmovsxbd %xmm0, %zmm0 {%k1} {z} 
;KNL-NEXT:  retq
define <16 x i32> @sext_16x8_to_16x32_mask(<16 x i8> %a , <16 x i1> %mask) nounwind readnone {
  %x   = sext <16 x i8> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

; KNL-LABEL: zext_16x8_to_16x32
; KNL: vpmovzxbd {{.*}}%zmm
; KNL: ret
define <16 x i32> @zext_16x8_to_16x32(<16 x i8> %i) nounwind readnone {
  %x = zext <16 x i8> %i to <16 x i32>
  ret <16 x i32> %x
}

; KNL-LABEL: sext_16x8_to_16x32
; KNL: vpmovsxbd {{.*}}%zmm
; KNL: ret
define <16 x i32> @sext_16x8_to_16x32(<16 x i8> %i) nounwind readnone {
  %x = sext <16 x i8> %i to <16 x i32>
  ret <16 x i32> %x
}

;SKX-LABEL: zext_2x8mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxbq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @zext_2x8mem_to_2x64(<2 x i8> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i8>,<2 x i8> *%i,align 1
  %x   = zext <2 x i8> %a to <2 x i64>
  %ret = select <2 x  i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}
;SKX-LABEL: sext_2x8mem_to_2x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxbq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @sext_2x8mem_to_2x64mask(<2 x i8> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i8>,<2 x i8> *%i,align 1
  %x   = sext <2 x i8> %a to <2 x i64>
  %ret = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}
;SKX-LABEL: sext_2x8mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxbq (%rdi), %xmm0   
;SKX-NEXT:  retq
define <2 x i64> @sext_2x8mem_to_2x64(<2 x i8> *%i) nounwind readnone {
  %a   = load <2 x i8>,<2 x i8> *%i,align 1
  %x   = sext <2 x i8> %a to <2 x i64>
  ret <2 x i64> %x
}

;SKX-LABEL: zext_4x8mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxbq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @zext_4x8mem_to_4x64(<4 x i8> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i8>,<4 x i8> *%i,align 1
  %x   = zext <4 x i8> %a to <4 x i64>
  %ret = select <4 x  i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x8mem_to_4x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxbq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @sext_4x8mem_to_4x64mask(<4 x i8> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i8>,<4 x i8> *%i,align 1
  %x   = sext <4 x i8> %a to <4 x i64>
  %ret = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x8mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxbq (%rdi), %ymm0   
;SKX-NEXT:  retq
define <4 x i64> @sext_4x8mem_to_4x64(<4 x i8> *%i) nounwind readnone {
  %a   = load <4 x i8>,<4 x i8> *%i,align 1
  %x   = sext <4 x i8> %a to <4 x i64>
  ret <4 x i64> %x
}

;KNL-LABEL: zext_8x8mem_to_8x64:
;KNL:       vpmovzxbq (%rdi), %zmm0 {%k1} {z} 
;KNL-NEXT:  retq
define <8 x i64> @zext_8x8mem_to_8x64(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = zext <8 x i8> %a to <8 x i64>
  %ret = select <8 x  i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;KNL-LABEL: sext_8x8mem_to_8x64mask:
;KNL:       vpmovsxbq (%rdi), %zmm0 {%k1} {z} 
;KNL-NEXT:  retq
define <8 x i64> @sext_8x8mem_to_8x64mask(<8 x i8> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = sext <8 x i8> %a to <8 x i64>
  %ret = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;KNL-LABEL: sext_8x8mem_to_8x64:
;KNL:       vpmovsxbq (%rdi), %zmm0   
;KNL-NEXT:  retq
define <8 x i64> @sext_8x8mem_to_8x64(<8 x i8> *%i) nounwind readnone {
  %a   = load <8 x i8>,<8 x i8> *%i,align 1
  %x   = sext <8 x i8> %a to <8 x i64>
  ret <8 x i64> %x
}

;SKX-LABEL: zext_4x16mem_to_4x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwd (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i32> @zext_4x16mem_to_4x32(<4 x i16> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = zext <4 x i16> %a to <4 x i32>
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

;SKX-LABEL: sext_4x16mem_to_4x32mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwd (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i32> @sext_4x16mem_to_4x32mask(<4 x i16> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = sext <4 x i16> %a to <4 x i32>
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

;SKX-LABEL: sext_4x16mem_to_4x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwd (%rdi), %xmm0   
;SKX-NEXT:  retq
define <4 x i32> @sext_4x16mem_to_4x32(<4 x i16> *%i) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = sext <4 x i16> %a to <4 x i32>
  ret <4 x i32> %x
}


;SKX-LABEL: zext_8x16mem_to_8x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwd (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i32> @zext_8x16mem_to_8x32(<8 x i16> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = zext <8 x i16> %a to <8 x i32>
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %ret
}

;SKX-LABEL: sext_8x16mem_to_8x32mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwd (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i32> @sext_8x16mem_to_8x32mask(<8 x i16> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = sext <8 x i16> %a to <8 x i32>
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %ret
}

;SKX-LABEL: sext_8x16mem_to_8x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwd (%rdi), %ymm0   
;SKX-NEXT:  retq
define <8 x i32> @sext_8x16mem_to_8x32(<8 x i16> *%i) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = sext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %x
}

;SKX-LABEL: zext_8x16_to_8x32mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm1, %k1
;SKX-NEXT:  vpmovzxwd %xmm0, %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i32> @zext_8x16_to_8x32mask(<8 x i16> %a , <8 x i1> %mask) nounwind readnone {
  %x   = zext <8 x i16> %a to <8 x i32>
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %ret
}

;SKX-LABEL: zext_8x16_to_8x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovzxwd %xmm0, %ymm0    
;SKX-NEXT:  retq
define <8 x i32> @zext_8x16_to_8x32(<8 x i16> %a ) nounwind readnone {
  %x   = zext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %x
}

;SKX-LABEL: zext_16x16mem_to_16x32:
;KNL-LABEL: zext_16x16mem_to_16x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwd (%rdi), %zmm0 {%k1} {z} 
;KNL:       vpmovzxwd (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <16 x i32> @zext_16x16mem_to_16x32(<16 x i16> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i16>,<16 x i16> *%i,align 1
  %x   = zext <16 x i16> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;SKX-LABEL: sext_16x16mem_to_16x32mask:
;KNL-LABEL: sext_16x16mem_to_16x32mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwd (%rdi), %zmm0 {%k1} {z} 
;KNL:       vpmovsxwd (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <16 x i32> @sext_16x16mem_to_16x32mask(<16 x i16> *%i , <16 x i1> %mask) nounwind readnone {
  %a   = load <16 x i16>,<16 x i16> *%i,align 1
  %x   = sext <16 x i16> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;SKX-LABEL: sext_16x16mem_to_16x32:
;KNL-LABEL: sext_16x16mem_to_16x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwd (%rdi), %zmm0   
;KNL:       vpmovsxwd (%rdi), %zmm0   
;SKX-NEXT:  retq
define <16 x i32> @sext_16x16mem_to_16x32(<16 x i16> *%i) nounwind readnone {
  %a   = load <16 x i16>,<16 x i16> *%i,align 1
  %x   = sext <16 x i16> %a to <16 x i32>
  ret <16 x i32> %x
}
;SKX-LABEL: zext_16x16_to_16x32mask:
;KNL-LABEL: zext_16x16_to_16x32mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovb2m  %xmm1, %k1
;SKX-NEXT:  vpmovzxwd %ymm0, %zmm0 {%k1} {z} 
;KNL:       vpmovzxwd %ymm0, %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <16 x i32> @zext_16x16_to_16x32mask(<16 x i16> %a , <16 x i1> %mask) nounwind readnone {
  %x   = zext <16 x i16> %a to <16 x i32>
  %ret = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> zeroinitializer
  ret <16 x i32> %ret
}

;SKX-LABEL: zext_16x16_to_16x32:
;KNL-LABEL: zext_16x16_to_16x32:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovzxwd %ymm0, %zmm0    
;KNL:       vpmovzxwd %ymm0, %zmm0    
;SKX-NEXT:  retq
define <16 x i32> @zext_16x16_to_16x32(<16 x i16> %a ) nounwind readnone {
  %x   = zext <16 x i16> %a to <16 x i32>
  ret <16 x i32> %x
}

;SKX-LABEL: zext_2x16mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @zext_2x16mem_to_2x64(<2 x i16> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i16>,<2 x i16> *%i,align 1
  %x   = zext <2 x i16> %a to <2 x i64>
  %ret = select <2 x  i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}

;SKX-LABEL: sext_2x16mem_to_2x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @sext_2x16mem_to_2x64mask(<2 x i16> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i16>,<2 x i16> *%i,align 1
  %x   = sext <2 x i16> %a to <2 x i64>
  %ret = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}

;SKX-LABEL: sext_2x16mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwq (%rdi), %xmm0   
;SKX-NEXT:  retq
define <2 x i64> @sext_2x16mem_to_2x64(<2 x i16> *%i) nounwind readnone {
  %a   = load <2 x i16>,<2 x i16> *%i,align 1
  %x   = sext <2 x i16> %a to <2 x i64>
  ret <2 x i64> %x
}

;SKX-LABEL: zext_4x16mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @zext_4x16mem_to_4x64(<4 x i16> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = zext <4 x i16> %a to <4 x i64>
  %ret = select <4 x  i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x16mem_to_4x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @sext_4x16mem_to_4x64mask(<4 x i16> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = sext <4 x i16> %a to <4 x i64>
  %ret = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x16mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwq (%rdi), %ymm0   
;SKX-NEXT:  retq
define <4 x i64> @sext_4x16mem_to_4x64(<4 x i16> *%i) nounwind readnone {
  %a   = load <4 x i16>,<4 x i16> *%i,align 1
  %x   = sext <4 x i16> %a to <4 x i64>
  ret <4 x i64> %x
}

;SKX-LABEL: zext_8x16mem_to_8x64:
;KNL-LABEL: zext_8x16mem_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxwq (%rdi), %zmm0 {%k1} {z} 
;KNL:       vpmovzxwq (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @zext_8x16mem_to_8x64(<8 x i16> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = zext <8 x i16> %a to <8 x i64>
  %ret = select <8 x  i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;SKX-LABEL: sext_8x16mem_to_8x64mask:
;KNL-LABEL: sext_8x16mem_to_8x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxwq (%rdi), %zmm0 {%k1} {z} 
;KNL:       vpmovsxwq (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @sext_8x16mem_to_8x64mask(<8 x i16> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = sext <8 x i16> %a to <8 x i64>
  %ret = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;SKX-LABEL: sext_8x16mem_to_8x64:
;KNL-LABEL: sext_8x16mem_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxwq (%rdi), %zmm0   
;KNL:       vpmovsxwq (%rdi), %zmm0   
;SKX-NEXT:  retq
define <8 x i64> @sext_8x16mem_to_8x64(<8 x i16> *%i) nounwind readnone {
  %a   = load <8 x i16>,<8 x i16> *%i,align 1
  %x   = sext <8 x i16> %a to <8 x i64>
  ret <8 x i64> %x
}

;SKX-LABEL: zext_8x16_to_8x64mask:
;KNL-LABEL: zext_8x16_to_8x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm1, %k1
;SKX-NEXT:  vpmovzxwq %xmm0, %zmm0 {%k1} {z} 
;KNL:       vpmovzxwq %xmm0, %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @zext_8x16_to_8x64mask(<8 x i16> %a , <8 x i1> %mask) nounwind readnone {
  %x   = zext <8 x i16> %a to <8 x i64>
  %ret = select <8 x  i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;SKX-LABEL: zext_8x16_to_8x64:
;KNL-LABEL: zext_8x16_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovzxwq %xmm0, %zmm0    
;KNL:       vpmovzxwq %xmm0, %zmm0    
;SKX-NEXT:  retq
; KNL: ret
define <8 x i64> @zext_8x16_to_8x64(<8 x i16> %a) nounwind readnone {
  %ret   = zext <8 x i16> %a to <8 x i64>
  ret <8 x i64> %ret
}

;SKX-LABEL: zext_2x32mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxdq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @zext_2x32mem_to_2x64(<2 x i32> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i32>,<2 x i32> *%i,align 1
  %x   = zext <2 x i32> %a to <2 x i64>
  %ret = select <2 x  i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}

;SKX-LABEL: sext_2x32mem_to_2x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovq2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxdq (%rdi), %xmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <2 x i64> @sext_2x32mem_to_2x64mask(<2 x i32> *%i , <2 x i1> %mask) nounwind readnone {
  %a   = load <2 x i32>,<2 x i32> *%i,align 1
  %x   = sext <2 x i32> %a to <2 x i64>
  %ret = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> zeroinitializer
  ret <2 x i64> %ret
}

;SKX-LABEL: sext_2x32mem_to_2x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxdq (%rdi), %xmm0   
;SKX-NEXT:  retq
define <2 x i64> @sext_2x32mem_to_2x64(<2 x i32> *%i) nounwind readnone {
  %a   = load <2 x i32>,<2 x i32> *%i,align 1
  %x   = sext <2 x i32> %a to <2 x i64>
  ret <2 x i64> %x
}

;SKX-LABEL: zext_4x32mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxdq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @zext_4x32mem_to_4x64(<4 x i32> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i32>,<4 x i32> *%i,align 1
  %x   = zext <4 x i32> %a to <4 x i64>
  %ret = select <4 x  i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x32mem_to_4x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxdq (%rdi), %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @sext_4x32mem_to_4x64mask(<4 x i32> *%i , <4 x i1> %mask) nounwind readnone {
  %a   = load <4 x i32>,<4 x i32> *%i,align 1
  %x   = sext <4 x i32> %a to <4 x i64>
  %ret = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: sext_4x32mem_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxdq (%rdi), %ymm0   
;SKX-NEXT:  retq
define <4 x i64> @sext_4x32mem_to_4x64(<4 x i32> *%i) nounwind readnone {
  %a   = load <4 x i32>,<4 x i32> *%i,align 1
  %x   = sext <4 x i32> %a to <4 x i64>
  ret <4 x i64> %x
}

;SKX-LABEL: sext_4x32_to_4x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxdq %xmm0, %ymm0    
;SKX-NEXT:  retq
define <4 x i64> @sext_4x32_to_4x64(<4 x i32> %a) nounwind readnone {
  %x   = sext <4 x i32> %a to <4 x i64>
  ret <4 x i64> %x
}

;SKX-LABEL: zext_4x32_to_4x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovd2m  %xmm1, %k1
;SKX-NEXT:  vpmovzxdq %xmm0, %ymm0 {%k1} {z} 
;SKX-NEXT:  retq
define <4 x i64> @zext_4x32_to_4x64mask(<4 x i32> %a , <4 x i1> %mask) nounwind readnone {
  %x   = zext <4 x i32> %a to <4 x i64>
  %ret = select <4 x  i1> %mask, <4 x i64> %x, <4 x i64> zeroinitializer
  ret <4 x i64> %ret
}

;SKX-LABEL: zext_8x32mem_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovzxdq (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @zext_8x32mem_to_8x64(<8 x i32> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i32>,<8 x i32> *%i,align 1
  %x   = zext <8 x i32> %a to <8 x i64>
  %ret = select <8 x  i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;SKX-LABEL: sext_8x32mem_to_8x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm0, %k1
;SKX-NEXT:  vpmovsxdq (%rdi), %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @sext_8x32mem_to_8x64mask(<8 x i32> *%i , <8 x i1> %mask) nounwind readnone {
  %a   = load <8 x i32>,<8 x i32> *%i,align 1
  %x   = sext <8 x i32> %a to <8 x i64>
  %ret = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}

;SKX-LABEL: sext_8x32mem_to_8x64:
;KNL-LABEL: sext_8x32mem_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxdq (%rdi), %zmm0   
;KNL:       vpmovsxdq (%rdi), %zmm0   
;SKX-NEXT:  retq
define <8 x i64> @sext_8x32mem_to_8x64(<8 x i32> *%i) nounwind readnone {
  %a   = load <8 x i32>,<8 x i32> *%i,align 1
  %x   = sext <8 x i32> %a to <8 x i64>
  ret <8 x i64> %x
}

;SKX-LABEL: sext_8x32_to_8x64:
;KNL-LABEL: sext_8x32_to_8x64:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovsxdq %ymm0, %zmm0    
;KNL:       vpmovsxdq %ymm0, %zmm0    
;SKX-NEXT:  retq
define <8 x i64> @sext_8x32_to_8x64(<8 x i32> %a) nounwind readnone {
  %x   = sext <8 x i32> %a to <8 x i64>
  ret <8 x i64> %x
}

;SKX-LABEL: zext_8x32_to_8x64mask:
;KNL-LABEL: zext_8x32_to_8x64mask:
;SKX:       ## BB#0:
;SKX-NEXT:  vpmovw2m  %xmm1, %k1
;SKX-NEXT:  vpmovzxdq %ymm0, %zmm0 {%k1} {z} 
;KNL:       vpmovzxdq %ymm0, %zmm0 {%k1} {z} 
;SKX-NEXT:  retq
define <8 x i64> @zext_8x32_to_8x64mask(<8 x i32> %a , <8 x i1> %mask) nounwind readnone {
  %x   = zext <8 x i32> %a to <8 x i64>
  %ret = select <8 x  i1> %mask, <8 x i64> %x, <8 x i64> zeroinitializer
  ret <8 x i64> %ret
}
;KNL-LABEL: fptrunc_test
;KNL: vcvtpd2ps {{.*}}%zmm
;KNL: ret
define <8 x float> @fptrunc_test(<8 x double> %a) nounwind readnone {
  %b = fptrunc <8 x double> %a to <8 x float>
  ret <8 x float> %b
}

;KNL-LABEL: fpext_test
;KNL: vcvtps2pd {{.*}}%zmm
;KNL: ret
define <8 x double> @fpext_test(<8 x float> %a) nounwind readnone {
  %b = fpext <8 x float> %a to <8 x double>
  ret <8 x double> %b
}

; KNL-LABEL: zext_16i1_to_16xi32
; KNL: vpbroadcastd LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; KNL: ret
define   <16 x i32> @zext_16i1_to_16xi32(i16 %b) {
  %a = bitcast i16 %b to <16 x i1>
  %c = zext <16 x i1> %a to <16 x i32>
  ret <16 x i32> %c
}

; KNL-LABEL: zext_8i1_to_8xi64
; KNL: vpbroadcastq LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; KNL: ret
define   <8 x i64> @zext_8i1_to_8xi64(i8 %b) {
  %a = bitcast i8 %b to <8 x i1>
  %c = zext <8 x i1> %a to <8 x i64>
  ret <8 x i64> %c
}

; KNL-LABEL: trunc_16i8_to_16i1
; KNL: vpmovsxbd
; KNL: vpandd
; KNL: vptestmd
; KNL: ret
; SKX-LABEL: trunc_16i8_to_16i1
; SKX: vpmovb2m %xmm
define i16 @trunc_16i8_to_16i1(<16 x i8> %a) {
  %mask_b = trunc <16 x i8>%a to <16 x i1>
  %mask = bitcast <16 x i1> %mask_b to i16
  ret i16 %mask
}

; KNL-LABEL: trunc_16i32_to_16i1
; KNL: vpandd
; KNL: vptestmd
; KNL: ret
; SKX-LABEL: trunc_16i32_to_16i1
; SKX: vpmovd2m %zmm
define i16 @trunc_16i32_to_16i1(<16 x i32> %a) {
  %mask_b = trunc <16 x i32>%a to <16 x i1>
  %mask = bitcast <16 x i1> %mask_b to i16
  ret i16 %mask
}

; SKX-LABEL: trunc_4i32_to_4i1
; SKX: vpmovd2m        %xmm
; SKX: kandw
; SKX: vpmovm2d
define <4 x i32> @trunc_4i32_to_4i1(<4 x i32> %a, <4 x i32> %b) {
  %mask_a = trunc <4 x i32>%a to <4 x i1>
  %mask_b = trunc <4 x i32>%b to <4 x i1>
  %a_and_b = and <4 x i1>%mask_a, %mask_b
  %res = sext <4 x i1>%a_and_b to <4 x i32>
  ret <4 x i32>%res
}

; KNL-LABEL: trunc_8i16_to_8i1
; KNL: vpmovsxwq
; KNL: vpandq LCP{{.*}}(%rip){1to8}
; KNL: vptestmq
; KNL: ret

; SKX-LABEL: trunc_8i16_to_8i1
; SKX: vpmovw2m %xmm
define i8 @trunc_8i16_to_8i1(<8 x i16> %a) {
  %mask_b = trunc <8 x i16>%a to <8 x i1>
  %mask = bitcast <8 x i1> %mask_b to i8
  ret i8 %mask
}

; KNL-LABEL: sext_8i1_8i32
; KNL: vpbroadcastq  LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; SKX: vpmovm2d
; KNL: ret
define <8 x i32> @sext_8i1_8i32(<8 x i32> %a1, <8 x i32> %a2) nounwind {
  %x = icmp slt <8 x i32> %a1, %a2
  %x1 = xor <8 x i1>%x, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
  %y = sext <8 x i1> %x1 to <8 x i32>
  ret <8 x i32> %y
}


; KNL-LABEL: trunc_i32_to_i1
; KNL: movw    $-4, %ax
; KNL: kmovw   %eax, %k1
; KNL: korw
define i16 @trunc_i32_to_i1(i32 %a) {
  %a_i = trunc i32 %a to i1
  %maskv = insertelement <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i1 %a_i, i32 0
  %res = bitcast <16 x i1> %maskv to i16
  ret i16 %res
}

; KNL-LABEL: sext_8i1_8i16
; SKX: vpmovm2w
; KNL: ret
define <8 x i16> @sext_8i1_8i16(<8 x i32> %a1, <8 x i32> %a2) nounwind {
  %x = icmp slt <8 x i32> %a1, %a2
  %y = sext <8 x i1> %x to <8 x i16>
  ret <8 x i16> %y
}

; KNL-LABEL: sext_16i1_16i32
; SKX: vpmovm2d
; KNL: ret
define <16 x i32> @sext_16i1_16i32(<16 x i32> %a1, <16 x i32> %a2) nounwind {
  %x = icmp slt <16 x i32> %a1, %a2
  %y = sext <16 x i1> %x to <16 x i32>
  ret <16 x i32> %y
}

; KNL-LABEL: sext_8i1_8i64
; SKX: vpmovm2q
; KNL: ret
define <8 x i64> @sext_8i1_8i64(<8 x i32> %a1, <8 x i32> %a2) nounwind {
  %x = icmp slt <8 x i32> %a1, %a2
  %y = sext <8 x i1> %x to <8 x i64>
  ret <8 x i64> %y
}

; KNL-LABEL: @extload_v8i64
; KNL: vpmovsxbq
define void @extload_v8i64(<8 x i8>* %a, <8 x i64>* %res) {
  %sign_load = load <8 x i8>, <8 x i8>* %a
  %c = sext <8 x i8> %sign_load to <8 x i64>
  store <8 x i64> %c, <8 x i64>* %res
  ret void
}

;SKX-LABEL: test21:
;SKX:       vmovdqu16 %zmm0, %zmm3 {%k1}
;SKX-NEXT:  kshiftrq  $32, %k1, %k1
;SKX-NEXT:  vmovdqu16 %zmm1, %zmm2 {%k1}
define <64 x i16> @test21(<64 x i16> %x , <64 x i1> %mask) nounwind readnone {
  %ret = select <64 x i1> %mask, <64 x i16> %x, <64 x i16> zeroinitializer
  ret <64 x i16> %ret
}  


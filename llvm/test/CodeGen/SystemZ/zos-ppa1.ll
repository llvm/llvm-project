; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s

define i32 @leaf(i32 signext %arg) {
; CHECK: * XPLINK Routine Layout Entry
; CHECK-NEXT: L#EPM_leaf_0 DS 0H                                                              
; CHECK-NEXT: * Eyecatcher 0x00C300C500C500
; CHECK-NEXT:  DC XL7'00C300C500C500'                                                         
; CHECK-NEXT: * Mark Type C'1'
; CHECK-NEXT:  DC XL1'F1'                                                                     
; CHECK-NEXT: * Offset to PPA1
; CHECK-NEXT:  DC AD(L#PPA1_leaf_0-L#EPM_leaf_0)                                              
; CHECK-NEXT: * DSA Size 0x0
; CHECK-NEXT: * Entry Flags
; CHECK-NEXT: *   Bit 1: 1 = Leaf function
; CHECK-NEXT: *   Bit 2: 0 = Does not use alloca
; CHECK-NEXT:  DC XL4'00000008'                                                               
; --- Code ---
; CHECK: stdin#C CSECT                                                                   
; CHECK-NEXT: C_CODE64 CATTR                                                                  
; CHECK-NEXT: * PPA1
; CHECK-NEXT: L#PPA1_leaf_0 DS 0H                                                             
; CHECK-NEXT: * Version
; CHECK-NEXT:  DC XL1'02'                                                                     
; CHECK-NEXT: * LE Signature X'CE'
; CHECK-NEXT:  DC XL1'CE'                                                                     
; CHECK-NEXT: * Saved GPR Mask
; CHECK-NEXT:  DC XL2'0000'                                                                   
; CHECK-NEXT: * Offset to PPA2
; CHECK-NEXT:  DC AD(L#PPA2-L#PPA1_leaf_0)                                                    
; CHECK-NEXT: * PPA1 Flags 1
; CHECK-NEXT: *   Bit 0: 1 = 64-bit DSA
; CHECK-NEXT:  DC XL1'80'                                                                     
; CHECK-NEXT: * PPA1 Flags 2
; CHECK-NEXT: *   Bit 0: 1 = External procedure
; CHECK-NEXT: *   Bit 3: 0 = STACKPROTECT is not enabled
; CHECK-NEXT:  DC XL1'80'                                                                     
; CHECK-NEXT: * PPA1 Flags 3
; CHECK-NEXT:  DC XL1'00'                                                                     
; CHECK-NEXT: * PPA1 Flags 4
; CHECK-NEXT: *   Bit 7: 1 = Name Length and Name
; CHECK-NEXT:  DC XL1'81'                                                                     
; CHECK-NEXT: * Length/4 of Parms
; CHECK-NEXT:  DC XL2'0002'                                                                   
; CHECK-NEXT: * Length/2 of Prolog 
; CHECK-NEXT:  DC XL1'00'                                                                     
; CHECK-NEXT: * Alloca Reg + Offset/2 to SP Update
; CHECK-NEXT: *   Bit 0-3: Register R0
; CHECK-NEXT: *   Bit 4-8: Offset 
; CHECK-NEXT:  DC XL1'0'                                                                      
; CHECK-NEXT: * Length of Code
; CHECK-NEXT:  DC AD(L#func_end0-L#EPM_leaf_0)                                                
; CHECK-NEXT: * Length of Name
; CHECK-NEXT:  DC XL2'0004'                                                                   
; CHECK-NEXT: * Name of Function
; CHECK-NEXT:  DC XL4'93858186'                                                               
; CHECK-NEXT:  DC AD(L#EPM_leaf_0-L#PPA1_leaf_0)                                              
entry:
  %sum = add i32 1, %arg
  ret i32 %sum
}

define i32 @nonleaf(i32 signext %arg) {
; CHECK: L#EPM_nonleaf_0 DS 0H                                                           
; CHECK-NEXT: * Eyecatcher 0x00C300C500C500
; CHECK-NEXT:  DC XL7'00C300C500C500'                                                         
; CHECK-NEXT: * Mark Type C'1'
; CHECK-NEXT:  DC XL1'F1'                                                                     
; CHECK-NEXT: * Offset to PPA1
; CHECK-NEXT:  DC AD(L#PPA1_nonleaf_0-L#EPM_nonleaf_0)                                        
; CHECK-NEXT: * DSA Size 0xc0
; CHECK-NEXT: * Entry Flags
; CHECK-NEXT: *   Bit 1: 0 = Non-leaf function
; CHECK-NEXT: *   Bit 2: 0 = Does not use alloca
; CHECK-NEXT:  DC XL4'000000C0'                                                               
; --- Code ---
; CHECK: stdin#C CSECT                                                                   
; CHECK-NEXT: C_CODE64 CATTR                                                                  
; CHECK-NEXT: * PPA1
; CHECK-NEXT: L#PPA1_nonleaf_0 DS 0H                                                          
; CHECK-NEXT: * Version
; CHECK-NEXT:  DC XL1'02'                                                                     
; CHECK-NEXT: * LE Signature X'CE'
; CHECK-NEXT:  DC XL1'CE'                                                                     
; CHECK-NEXT: * Saved GPR Mask
; CHECK-NEXT:  DC XL2'0300'                                                                   
; CHECK-NEXT: * Offset to PPA2
; CHECK-NEXT:  DC AD(L#PPA2-L#PPA1_nonleaf_0)                                                 
; CHECK-NEXT: * PPA1 Flags 1
; CHECK-NEXT: *   Bit 0: 1 = 64-bit DSA
; CHECK-NEXT:  DC XL1'80'                                                                     
; CHECK-NEXT: * PPA1 Flags 2
; CHECK-NEXT: *   Bit 0: 1 = External procedure
; CHECK-NEXT: *   Bit 3: 0 = STACKPROTECT is not enabled
; CHECK-NEXT:  DC XL1'80'                                                                     
; CHECK-NEXT: * PPA1 Flags 3
; CHECK-NEXT:  DC XL1'00'                                                                     
; CHECK-NEXT: * PPA1 Flags 4
; CHECK-NEXT: *   Bit 7: 1 = Name Length and Name
; CHECK-NEXT:  DC XL1'81'                                                                     
; CHECK-NEXT: * Length/4 of Parms
; CHECK-NEXT:  DC XL2'0002'                                                                   
; CHECK-NEXT: L#tmp0 EQU L#end_of_prologue0-nonleaf
; CHECK-NEXT: * Length/2 of Prolog 
; CHECK-NEXT:  DC AD(L#tmp0/2)                                                                
; CHECK-NEXT: L#tmp1 EQU L#stack_update0-nonleaf
; CHECK-NEXT: * Alloca Reg + Offset/2 to SP Update
; CHECK-NEXT: *   Bit 0-3: Register R0
; CHECK-NEXT: *   Bit 4-8: Offset 
; CHECK-NEXT:  DC AD(L#tmp1/2),XL1'0'                                                         
; CHECK-NEXT: * Length of Code
; CHECK-NEXT:  DC AD(L#func_end2-L#EPM_nonleaf_0)                                             
; CHECK-NEXT: * Length of Name
; CHECK-NEXT:  DC XL2'0007'                                                                   
; CHECK-NEXT: * Name of Function
; CHECK-NEXT:  DC XL7'95969593858186'                                                         
; CHECK-NEXT:  DC AD(L#EPM_nonleaf_0-L#PPA1_nonleaf_0)                                        
entry:
  %res = call i32 @leaf(i32 %arg)
  ret i32 %res
}

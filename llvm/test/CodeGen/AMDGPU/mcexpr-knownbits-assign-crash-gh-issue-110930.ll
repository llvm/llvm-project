; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s

; Previously, this would hit an assertion on incompatible comparison between
; APInts due to BitWidth differences. This was due to assignment of DenseMap
; value using another value within that same DenseMap which results in a
; use-after-free if the assignment operator invokes a DenseMap growth.

; CHECK-LABEL: I_Quit:
; CHECK: .set .LI_Quit.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set .LI_Quit.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set .LI_Quit.numbered_sgpr, max(56, amdgpu.max_num_sgpr)
; CHECK: .set .LI_Quit.private_seg_size, 16
; CHECK: .set .LI_Quit.uses_vcc, 1
; CHECK: .set .LI_Quit.uses_flat_scratch, 1
; CHECK: .set .LI_Quit.has_dyn_sized_stack, 1
; CHECK: .set .LI_Quit.has_recursion, 1
; CHECK: .set .LI_Quit.has_indirect_call, 1
define void @I_Quit() {
  %fptr = load ptr, ptr null, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: P_RemoveMobj:
; CHECK: .set .LP_RemoveMobj.num_vgpr, 0
; CHECK: .set .LP_RemoveMobj.num_agpr, 0
; CHECK: .set .LP_RemoveMobj.numbered_sgpr, 32
; CHECK: .set .LP_RemoveMobj.private_seg_size, 0
; CHECK: .set .LP_RemoveMobj.uses_vcc, 0
; CHECK: .set .LP_RemoveMobj.uses_flat_scratch, 0
; CHECK: .set .LP_RemoveMobj.has_dyn_sized_stack, 0
; CHECK: .set .LP_RemoveMobj.has_recursion, 0
; CHECK: .set .LP_RemoveMobj.has_indirect_call, 0
define void @P_RemoveMobj() {
  ret void
}

; CHECK-LABEL: P_SpawnMobj:
; CHECK: .set .LP_SpawnMobj.num_vgpr, 0
; CHECK: .set .LP_SpawnMobj.num_agpr, 0
; CHECK: .set .LP_SpawnMobj.numbered_sgpr, 32
; CHECK: .set .LP_SpawnMobj.private_seg_size, 0
; CHECK: .set .LP_SpawnMobj.uses_vcc, 0
; CHECK: .set .LP_SpawnMobj.uses_flat_scratch, 0
; CHECK: .set .LP_SpawnMobj.has_dyn_sized_stack, 0
; CHECK: .set .LP_SpawnMobj.has_recursion, 0
; CHECK: .set .LP_SpawnMobj.has_indirect_call, 0
define void @P_SpawnMobj() {
  ret void
}

; CHECK-LABEL: G_PlayerReborn:
; CHECK: .set .LG_PlayerReborn.num_vgpr, 0
; CHECK: .set .LG_PlayerReborn.num_agpr, 0
; CHECK: .set .LG_PlayerReborn.numbered_sgpr, 32
; CHECK: .set .LG_PlayerReborn.private_seg_size, 0
; CHECK: .set .LG_PlayerReborn.uses_vcc, 0
; CHECK: .set .LG_PlayerReborn.uses_flat_scratch, 0
; CHECK: .set .LG_PlayerReborn.has_dyn_sized_stack, 0
; CHECK: .set .LG_PlayerReborn.has_recursion, 0
; CHECK: .set .LG_PlayerReborn.has_indirect_call, 0
define void @G_PlayerReborn() {
  ret void
}

; CHECK-LABEL: P_SetThingPosition:
; CHECK: .set .LP_SetThingPosition.num_vgpr, 0
; CHECK: .set .LP_SetThingPosition.num_agpr, 0
; CHECK: .set .LP_SetThingPosition.numbered_sgpr, 32
; CHECK: .set .LP_SetThingPosition.private_seg_size, 0
; CHECK: .set .LP_SetThingPosition.uses_vcc, 0
; CHECK: .set .LP_SetThingPosition.uses_flat_scratch, 0
; CHECK: .set .LP_SetThingPosition.has_dyn_sized_stack, 0
; CHECK: .set .LP_SetThingPosition.has_recursion, 0
; CHECK: .set .LP_SetThingPosition.has_indirect_call, 0
define void @P_SetThingPosition() {
  ret void
}

; CHECK-LABEL: P_SetupPsprites:
; CHECK: .set .LP_SetupPsprites.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set .LP_SetupPsprites.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set .LP_SetupPsprites.numbered_sgpr, max(56, amdgpu.max_num_sgpr)
; CHECK: .set .LP_SetupPsprites.private_seg_size, 16
; CHECK: .set .LP_SetupPsprites.uses_vcc, 1
; CHECK: .set .LP_SetupPsprites.uses_flat_scratch, 1
; CHECK: .set .LP_SetupPsprites.has_dyn_sized_stack, 1
; CHECK: .set .LP_SetupPsprites.has_recursion, 1
; CHECK: .set .LP_SetupPsprites.has_indirect_call, 1
define void @P_SetupPsprites(ptr addrspace(1) %i) {
  %fptr = load ptr, ptr addrspace(1) %i, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: HU_Start:
; CHECK: .set .LHU_Start.num_vgpr, 0
; CHECK: .set .LHU_Start.num_agpr, 0
; CHECK: .set .LHU_Start.numbered_sgpr, 32
; CHECK: .set .LHU_Start.private_seg_size, 0
; CHECK: .set .LHU_Start.uses_vcc, 0
; CHECK: .set .LHU_Start.uses_flat_scratch, 0
; CHECK: .set .LHU_Start.has_dyn_sized_stack, 0
; CHECK: .set .LHU_Start.has_recursion, 0
; CHECK: .set .LHU_Start.has_indirect_call, 0
define void @HU_Start() {
  ret void
}

; CHECK-LABEL: P_SpawnPlayer:
; CHECK: .set .LP_SpawnPlayer.num_vgpr, max(43, .LG_PlayerReborn.num_vgpr, .LP_SetThingPosition.num_vgpr, .LP_SetupPsprites.num_vgpr, .LHU_Start.num_vgpr)
; CHECK: .set .LP_SpawnPlayer.num_agpr, max(0, .LG_PlayerReborn.num_agpr, .LP_SetThingPosition.num_agpr, .LP_SetupPsprites.num_agpr, .LHU_Start.num_agpr)
; CHECK: .set .LP_SpawnPlayer.numbered_sgpr, max(84, .LG_PlayerReborn.numbered_sgpr, .LP_SetThingPosition.numbered_sgpr, .LP_SetupPsprites.numbered_sgpr, .LHU_Start.numbered_sgpr)
; CHECK: .set .LP_SpawnPlayer.private_seg_size, 16+max(.LG_PlayerReborn.private_seg_size, .LP_SetThingPosition.private_seg_size, .LP_SetupPsprites.private_seg_size, .LHU_Start.private_seg_size)
; CHECK: .set .LP_SpawnPlayer.uses_vcc, or(1, .LG_PlayerReborn.uses_vcc, .LP_SetThingPosition.uses_vcc, .LP_SetupPsprites.uses_vcc, .LHU_Start.uses_vcc)
; CHECK: .set .LP_SpawnPlayer.uses_flat_scratch, or(0, .LG_PlayerReborn.uses_flat_scratch, .LP_SetThingPosition.uses_flat_scratch, .LP_SetupPsprites.uses_flat_scratch, .LHU_Start.uses_flat_scratch)
; CHECK: .set .LP_SpawnPlayer.has_dyn_sized_stack, or(0, .LG_PlayerReborn.has_dyn_sized_stack, .LP_SetThingPosition.has_dyn_sized_stack, .LP_SetupPsprites.has_dyn_sized_stack, .LHU_Start.has_dyn_sized_stack)
; CHECK: .set .LP_SpawnPlayer.has_recursion, or(1, .LG_PlayerReborn.has_recursion, .LP_SetThingPosition.has_recursion, .LP_SetupPsprites.has_recursion, .LHU_Start.has_recursion)
; CHECK: .set .LP_SpawnPlayer.has_indirect_call, or(0, .LG_PlayerReborn.has_indirect_call, .LP_SetThingPosition.has_indirect_call, .LP_SetupPsprites.has_indirect_call, .LHU_Start.has_indirect_call)
define void @P_SpawnPlayer() {
  call void @G_PlayerReborn()
  call void @P_SetThingPosition()
  call void @P_SetupPsprites(ptr addrspace(1) null)
  tail call void @HU_Start()
  ret void
}

; CHECK-LABEL: I_Error:
; CHECK: .set .LI_Error.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set .LI_Error.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set .LI_Error.numbered_sgpr, max(56, amdgpu.max_num_sgpr)
; CHECK: .set .LI_Error.private_seg_size, 16
; CHECK: .set .LI_Error.uses_vcc, 1
; CHECK: .set .LI_Error.uses_flat_scratch, 1
; CHECK: .set .LI_Error.has_dyn_sized_stack, 1
; CHECK: .set .LI_Error.has_recursion, 1
; CHECK: .set .LI_Error.has_indirect_call, 1
define void @I_Error(...) {
  %fptr = load ptr, ptr null, align 8
  call void %fptr()
  ret void
}

; CHECK-LABEL: G_DoReborn:
; CHECK: .set .LG_DoReborn.num_vgpr, max(44, .LP_RemoveMobj.num_vgpr, .LP_SpawnMobj.num_vgpr, .LP_SpawnPlayer.num_vgpr, .LI_Error.num_vgpr)
; CHECK: .set .LG_DoReborn.num_agpr, max(0, .LP_RemoveMobj.num_agpr, .LP_SpawnMobj.num_agpr, .LP_SpawnPlayer.num_agpr, .LI_Error.num_agpr)
; CHECK: .set .LG_DoReborn.numbered_sgpr, max(104, .LP_RemoveMobj.numbered_sgpr, .LP_SpawnMobj.numbered_sgpr, .LP_SpawnPlayer.numbered_sgpr, .LI_Error.numbered_sgpr)
; CHECK: .set .LG_DoReborn.private_seg_size, 32+max(.LP_RemoveMobj.private_seg_size, .LP_SpawnMobj.private_seg_size, .LP_SpawnPlayer.private_seg_size, .LI_Error.private_seg_size)
; CHECK: .set .LG_DoReborn.uses_vcc, or(1, .LP_RemoveMobj.uses_vcc, .LP_SpawnMobj.uses_vcc, .LP_SpawnPlayer.uses_vcc, .LI_Error.uses_vcc)
; CHECK: .set .LG_DoReborn.uses_flat_scratch, or(0, .LP_RemoveMobj.uses_flat_scratch, .LP_SpawnMobj.uses_flat_scratch, .LP_SpawnPlayer.uses_flat_scratch, .LI_Error.uses_flat_scratch)
; CHECK: .set .LG_DoReborn.has_dyn_sized_stack, or(0, .LP_RemoveMobj.has_dyn_sized_stack, .LP_SpawnMobj.has_dyn_sized_stack, .LP_SpawnPlayer.has_dyn_sized_stack, .LI_Error.has_dyn_sized_stack)
; CHECK: .set .LG_DoReborn.has_recursion, or(1, .LP_RemoveMobj.has_recursion, .LP_SpawnMobj.has_recursion, .LP_SpawnPlayer.has_recursion, .LI_Error.has_recursion)
; CHECK: .set .LG_DoReborn.has_indirect_call, or(0, .LP_RemoveMobj.has_indirect_call, .LP_SpawnMobj.has_indirect_call, .LP_SpawnPlayer.has_indirect_call, .LI_Error.has_indirect_call)
define void @G_DoReborn() {
  call void @P_RemoveMobj()
  call void @P_SpawnMobj()
  call void @P_SpawnPlayer()
  call void (...) @I_Error()
  ret void
}

; CHECK-LABEL: AM_Stop:
; CHECK: .set .LAM_Stop.num_vgpr, 0
; CHECK: .set .LAM_Stop.num_agpr, 0
; CHECK: .set .LAM_Stop.numbered_sgpr, 32
; CHECK: .set .LAM_Stop.private_seg_size, 0
; CHECK: .set .LAM_Stop.uses_vcc, 0
; CHECK: .set .LAM_Stop.uses_flat_scratch, 0
; CHECK: .set .LAM_Stop.has_dyn_sized_stack, 0
; CHECK: .set .LAM_Stop.has_recursion, 0
; CHECK: .set .LAM_Stop.has_indirect_call, 0
define void @AM_Stop() {
  ret void
}

; CHECK-LABEL: D_AdvanceDemo:
; CHECK: .set .LD_AdvanceDemo.num_vgpr, 0
; CHECK: .set .LD_AdvanceDemo.num_agpr, 0
; CHECK: .set .LD_AdvanceDemo.numbered_sgpr, 32
; CHECK: .set .LD_AdvanceDemo.private_seg_size, 0
; CHECK: .set .LD_AdvanceDemo.uses_vcc, 0
; CHECK: .set .LD_AdvanceDemo.uses_flat_scratch, 0
; CHECK: .set .LD_AdvanceDemo.has_dyn_sized_stack, 0
; CHECK: .set .LD_AdvanceDemo.has_recursion, 0
; CHECK: .set .LD_AdvanceDemo.has_indirect_call, 0
define void @D_AdvanceDemo() {
  ret void
}

; CHECK-LABEL: F_StartFinale:
; CHECK: .set .LF_StartFinale.num_vgpr, 0
; CHECK: .set .LF_StartFinale.num_agpr, 0
; CHECK: .set .LF_StartFinale.numbered_sgpr, 32
; CHECK: .set .LF_StartFinale.private_seg_size, 0
; CHECK: .set .LF_StartFinale.uses_vcc, 0
; CHECK: .set .LF_StartFinale.uses_flat_scratch, 0
; CHECK: .set .LF_StartFinale.has_dyn_sized_stack, 0
; CHECK: .set .LF_StartFinale.has_recursion, 0
; CHECK: .set .LF_StartFinale.has_indirect_call, 0
define void @F_StartFinale() {
  ret void
}

; CHECK-LABEL: F_Ticker:
; CHECK: .set .LF_Ticker.num_vgpr, 0
; CHECK: .set .LF_Ticker.num_agpr, 0
; CHECK: .set .LF_Ticker.numbered_sgpr, 32
; CHECK: .set .LF_Ticker.private_seg_size, 0
; CHECK: .set .LF_Ticker.uses_vcc, 0
; CHECK: .set .LF_Ticker.uses_flat_scratch, 0
; CHECK: .set .LF_Ticker.has_dyn_sized_stack, 0
; CHECK: .set .LF_Ticker.has_recursion, 0
; CHECK: .set .LF_Ticker.has_indirect_call, 0
define void @F_Ticker() {
  ret void
}

; CHECK-LABEL: G_CheckDemoStatus:
; CHECK: .set .LG_CheckDemoStatus.num_vgpr, max(43, .LI_Quit.num_vgpr, .LD_AdvanceDemo.num_vgpr, .LI_Error.num_vgpr)
; CHECK: .set .LG_CheckDemoStatus.num_agpr, max(0, .LI_Quit.num_agpr, .LD_AdvanceDemo.num_agpr, .LI_Error.num_agpr)
; CHECK: .set .LG_CheckDemoStatus.numbered_sgpr, max(84, .LI_Quit.numbered_sgpr, .LD_AdvanceDemo.numbered_sgpr, .LI_Error.numbered_sgpr)
; CHECK: .set .LG_CheckDemoStatus.private_seg_size, 32+max(.LI_Quit.private_seg_size, .LD_AdvanceDemo.private_seg_size, .LI_Error.private_seg_size)
; CHECK: .set .LG_CheckDemoStatus.uses_vcc, or(1, .LI_Quit.uses_vcc, .LD_AdvanceDemo.uses_vcc, .LI_Error.uses_vcc)
; CHECK: .set .LG_CheckDemoStatus.uses_flat_scratch, or(0, .LI_Quit.uses_flat_scratch, .LD_AdvanceDemo.uses_flat_scratch, .LI_Error.uses_flat_scratch)
; CHECK: .set .LG_CheckDemoStatus.has_dyn_sized_stack, or(0, .LI_Quit.has_dyn_sized_stack, .LD_AdvanceDemo.has_dyn_sized_stack, .LI_Error.has_dyn_sized_stack)
; CHECK: .set .LG_CheckDemoStatus.has_recursion, or(1, .LI_Quit.has_recursion, .LD_AdvanceDemo.has_recursion, .LI_Error.has_recursion)
; CHECK: .set .LG_CheckDemoStatus.has_indirect_call, or(0, .LI_Quit.has_indirect_call, .LD_AdvanceDemo.has_indirect_call, .LI_Error.has_indirect_call)
define i32 @G_CheckDemoStatus() {
  tail call void @I_Quit()
  tail call void @D_AdvanceDemo()
  call void (...) @I_Error()
  ret i32 0
}


; CHECK-LABEL: P_TempSaveGameFile:
; CHECK: .set .LP_TempSaveGameFile.num_vgpr, 2
; CHECK: .set .LP_TempSaveGameFile.num_agpr, 0
; CHECK: .set .LP_TempSaveGameFile.numbered_sgpr, 32
; CHECK: .set .LP_TempSaveGameFile.private_seg_size, 0
; CHECK: .set .LP_TempSaveGameFile.uses_vcc, 0
; CHECK: .set .LP_TempSaveGameFile.uses_flat_scratch, 0
; CHECK: .set .LP_TempSaveGameFile.has_dyn_sized_stack, 0
; CHECK: .set .LP_TempSaveGameFile.has_recursion, 0
; CHECK: .set .LP_TempSaveGameFile.has_indirect_call, 0
define ptr @P_TempSaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: P_SaveGameFile:
; CHECK: .set .LP_SaveGameFile.num_vgpr, 2
; CHECK: .set .LP_SaveGameFile.num_agpr, 0
; CHECK: .set .LP_SaveGameFile.numbered_sgpr, 32
; CHECK: .set .LP_SaveGameFile.private_seg_size, 0
; CHECK: .set .LP_SaveGameFile.uses_vcc, 0
; CHECK: .set .LP_SaveGameFile.uses_flat_scratch, 0
; CHECK: .set .LP_SaveGameFile.has_dyn_sized_stack, 0
; CHECK: .set .LP_SaveGameFile.has_recursion, 0
; CHECK: .set .LP_SaveGameFile.has_indirect_call, 0
define ptr @P_SaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: R_FlatNumForName:
; CHECK: .set .LR_FlatNumForName.num_vgpr, max(42, .LI_Error.num_vgpr)
; CHECK: .set .LR_FlatNumForName.num_agpr, max(0, .LI_Error.num_agpr)
; CHECK: .set .LR_FlatNumForName.numbered_sgpr, max(56, .LI_Error.numbered_sgpr)
; CHECK: .set .LR_FlatNumForName.private_seg_size, 16+max(.LI_Error.private_seg_size)
; CHECK: .set .LR_FlatNumForName.uses_vcc, or(1, .LI_Error.uses_vcc)
; CHECK: .set .LR_FlatNumForName.uses_flat_scratch, or(0, .LI_Error.uses_flat_scratch)
; CHECK: .set .LR_FlatNumForName.has_dyn_sized_stack, or(0, .LI_Error.has_dyn_sized_stack)
; CHECK: .set .LR_FlatNumForName.has_recursion, or(1, .LI_Error.has_recursion)
; CHECK: .set .LR_FlatNumForName.has_indirect_call, or(0, .LI_Error.has_indirect_call)
define i32 @R_FlatNumForName() {
  call void (...) @I_Error()
  unreachable
}

; CHECK-LABEL: R_TextureNumForName:
; CHECK: .set .LR_TextureNumForName.num_vgpr, max(42, .LR_FlatNumForName.num_vgpr)
; CHECK: .set .LR_TextureNumForName.num_agpr, max(0, .LR_FlatNumForName.num_agpr)
; CHECK: .set .LR_TextureNumForName.numbered_sgpr, max(56, .LR_FlatNumForName.numbered_sgpr)
; CHECK: .set .LR_TextureNumForName.private_seg_size, 16+max(.LR_FlatNumForName.private_seg_size)
; CHECK: .set .LR_TextureNumForName.uses_vcc, or(1, .LR_FlatNumForName.uses_vcc)
; CHECK: .set .LR_TextureNumForName.uses_flat_scratch, or(0, .LR_FlatNumForName.uses_flat_scratch)
; CHECK: .set .LR_TextureNumForName.has_dyn_sized_stack, or(0, .LR_FlatNumForName.has_dyn_sized_stack)
; CHECK: .set .LR_TextureNumForName.has_recursion, or(1, .LR_FlatNumForName.has_recursion)
; CHECK: .set .LR_TextureNumForName.has_indirect_call, or(0, .LR_FlatNumForName.has_indirect_call)
define i32 @R_TextureNumForName() {
  %ret = call i32 @R_FlatNumForName()
  ret i32 0
}

; CHECK-LABEL: G_Ticker:
; CHECK: .set .LG_Ticker.num_vgpr, max(47, .LG_DoReborn.num_vgpr, .LF_Ticker.num_vgpr, .LAM_Stop.num_vgpr, .LF_StartFinale.num_vgpr, .LD_AdvanceDemo.num_vgpr, .LR_FlatNumForName.num_vgpr, .LR_TextureNumForName.num_vgpr, .LP_TempSaveGameFile.num_vgpr, .LP_SaveGameFile.num_vgpr, .LI_Error.num_vgpr)
; CHECK: .set .LG_Ticker.num_agpr, max(0, .LG_DoReborn.num_agpr, .LF_Ticker.num_agpr, .LAM_Stop.num_agpr, .LF_StartFinale.num_agpr, .LD_AdvanceDemo.num_agpr, .LR_FlatNumForName.num_agpr, .LR_TextureNumForName.num_agpr, .LP_TempSaveGameFile.num_agpr, .LP_SaveGameFile.num_agpr, .LI_Error.num_agpr)
; CHECK: .set .LG_Ticker.numbered_sgpr, max(105, .LG_DoReborn.numbered_sgpr, .LF_Ticker.numbered_sgpr, .LAM_Stop.numbered_sgpr, .LF_StartFinale.numbered_sgpr, .LD_AdvanceDemo.numbered_sgpr, .LR_FlatNumForName.numbered_sgpr, .LR_TextureNumForName.numbered_sgpr, .LP_TempSaveGameFile.numbered_sgpr, .LP_SaveGameFile.numbered_sgpr, .LI_Error.numbered_sgpr)
; CHECK: .set .LG_Ticker.private_seg_size, 48+max(.LG_DoReborn.private_seg_size, .LF_Ticker.private_seg_size, .LAM_Stop.private_seg_size, .LF_StartFinale.private_seg_size, .LD_AdvanceDemo.private_seg_size, .LR_FlatNumForName.private_seg_size, .LR_TextureNumForName.private_seg_size, .LP_TempSaveGameFile.private_seg_size, .LP_SaveGameFile.private_seg_size, .LI_Error.private_seg_size)
; CHECK: .set .LG_Ticker.uses_vcc, or(1, .LG_DoReborn.uses_vcc, .LF_Ticker.uses_vcc, .LAM_Stop.uses_vcc, .LF_StartFinale.uses_vcc, .LD_AdvanceDemo.uses_vcc, .LR_FlatNumForName.uses_vcc, .LR_TextureNumForName.uses_vcc, .LP_TempSaveGameFile.uses_vcc, .LP_SaveGameFile.uses_vcc, .LI_Error.uses_vcc)
; CHECK: .set .LG_Ticker.uses_flat_scratch, or(0, .LG_DoReborn.uses_flat_scratch, .LF_Ticker.uses_flat_scratch, .LAM_Stop.uses_flat_scratch, .LF_StartFinale.uses_flat_scratch, .LD_AdvanceDemo.uses_flat_scratch, .LR_FlatNumForName.uses_flat_scratch, .LR_TextureNumForName.uses_flat_scratch, .LP_TempSaveGameFile.uses_flat_scratch, .LP_SaveGameFile.uses_flat_scratch, .LI_Error.uses_flat_scratch)
; CHECK: .set .LG_Ticker.has_dyn_sized_stack, or(0, .LG_DoReborn.has_dyn_sized_stack, .LF_Ticker.has_dyn_sized_stack, .LAM_Stop.has_dyn_sized_stack, .LF_StartFinale.has_dyn_sized_stack, .LD_AdvanceDemo.has_dyn_sized_stack, .LR_FlatNumForName.has_dyn_sized_stack, .LR_TextureNumForName.has_dyn_sized_stack, .LP_TempSaveGameFile.has_dyn_sized_stack, .LP_SaveGameFile.has_dyn_sized_stack, .LI_Error.has_dyn_sized_stack)
; CHECK: .set .LG_Ticker.has_recursion, or(1, .LG_DoReborn.has_recursion, .LF_Ticker.has_recursion, .LAM_Stop.has_recursion, .LF_StartFinale.has_recursion, .LD_AdvanceDemo.has_recursion, .LR_FlatNumForName.has_recursion, .LR_TextureNumForName.has_recursion, .LP_TempSaveGameFile.has_recursion, .LP_SaveGameFile.has_recursion, .LI_Error.has_recursion)
; CHECK: .set .LG_Ticker.has_indirect_call, or(0, .LG_DoReborn.has_indirect_call, .LF_Ticker.has_indirect_call, .LAM_Stop.has_indirect_call, .LF_StartFinale.has_indirect_call, .LD_AdvanceDemo.has_indirect_call, .LR_FlatNumForName.has_indirect_call, .LR_TextureNumForName.has_indirect_call, .LP_TempSaveGameFile.has_indirect_call, .LP_SaveGameFile.has_indirect_call, .LI_Error.has_indirect_call)
define void @G_Ticker() {
  call void @G_DoReborn()
  tail call void @F_Ticker()
  tail call void @AM_Stop()
  tail call void @F_StartFinale()
  tail call void @D_AdvanceDemo()
  %call.i.i449 = call i32 @R_FlatNumForName()
  %call9.i.i = call i32 @R_TextureNumForName()
  %call.i306 = tail call ptr @P_TempSaveGameFile()
  %call1.i307 = call ptr @P_SaveGameFile()
  call void (...) @I_Error()
  ret void
}

; CHECK-LABEL: RunTic:
; CHECK: .set .LRunTic.num_vgpr, max(47, .LG_CheckDemoStatus.num_vgpr, .LD_AdvanceDemo.num_vgpr, .LG_Ticker.num_vgpr)
; CHECK: .set .LRunTic.num_agpr, max(0, .LG_CheckDemoStatus.num_agpr, .LD_AdvanceDemo.num_agpr, .LG_Ticker.num_agpr)
; CHECK: .set .LRunTic.numbered_sgpr, max(105, .LG_CheckDemoStatus.numbered_sgpr, .LD_AdvanceDemo.numbered_sgpr, .LG_Ticker.numbered_sgpr)
; CHECK: .set .LRunTic.private_seg_size, 32+max(.LG_CheckDemoStatus.private_seg_size, .LD_AdvanceDemo.private_seg_size, .LG_Ticker.private_seg_size)
; CHECK: .set .LRunTic.uses_vcc, or(1, .LG_CheckDemoStatus.uses_vcc, .LD_AdvanceDemo.uses_vcc, .LG_Ticker.uses_vcc)
; CHECK: .set .LRunTic.uses_flat_scratch, or(0, .LG_CheckDemoStatus.uses_flat_scratch, .LD_AdvanceDemo.uses_flat_scratch, .LG_Ticker.uses_flat_scratch)
; CHECK: .set .LRunTic.has_dyn_sized_stack, or(0, .LG_CheckDemoStatus.has_dyn_sized_stack, .LD_AdvanceDemo.has_dyn_sized_stack, .LG_Ticker.has_dyn_sized_stack)
; CHECK: .set .LRunTic.has_recursion, or(1, .LG_CheckDemoStatus.has_recursion, .LD_AdvanceDemo.has_recursion, .LG_Ticker.has_recursion)
; CHECK: .set .LRunTic.has_indirect_call, or(0, .LG_CheckDemoStatus.has_indirect_call, .LD_AdvanceDemo.has_indirect_call, .LG_Ticker.has_indirect_call)
define void @RunTic() {
  %call5.i1 = call i32 @G_CheckDemoStatus()
  tail call void @D_AdvanceDemo()
  call void @G_Ticker()
  ret void
}

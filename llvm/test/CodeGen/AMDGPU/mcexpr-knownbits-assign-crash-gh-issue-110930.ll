; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s

; Previously, this would hit an assertion on incompatible comparison between
; APInts due to BitWidth differences. This was due to assignment of DenseMap
; value using another value within that same DenseMap which results in a
; use-after-free if the assignment operator invokes a DenseMap growth.

; CHECK-LABEL: I_Quit:
; CHECK: .set I_Quit.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set I_Quit.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set I_Quit.numbered_sgpr, max(48, amdgpu.max_num_sgpr)
; CHECK: .set I_Quit.private_seg_size, 16
; CHECK: .set I_Quit.uses_vcc, 1
; CHECK: .set I_Quit.uses_flat_scratch, 1
; CHECK: .set I_Quit.has_dyn_sized_stack, 1
; CHECK: .set I_Quit.has_recursion, 1
; CHECK: .set I_Quit.has_indirect_call, 1
define void @I_Quit() {
  %fptr = load ptr, ptr null, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: P_RemoveMobj:
; CHECK: .set P_RemoveMobj.num_vgpr, 0
; CHECK: .set P_RemoveMobj.num_agpr, 0
; CHECK: .set P_RemoveMobj.numbered_sgpr, 32
; CHECK: .set P_RemoveMobj.private_seg_size, 0
; CHECK: .set P_RemoveMobj.uses_vcc, 0
; CHECK: .set P_RemoveMobj.uses_flat_scratch, 0
; CHECK: .set P_RemoveMobj.has_dyn_sized_stack, 0
; CHECK: .set P_RemoveMobj.has_recursion, 0
; CHECK: .set P_RemoveMobj.has_indirect_call, 0
define void @P_RemoveMobj() {
  ret void
}

; CHECK-LABEL: P_SpawnMobj:
; CHECK: .set P_SpawnMobj.num_vgpr, 0
; CHECK: .set P_SpawnMobj.num_agpr, 0
; CHECK: .set P_SpawnMobj.numbered_sgpr, 32
; CHECK: .set P_SpawnMobj.private_seg_size, 0
; CHECK: .set P_SpawnMobj.uses_vcc, 0
; CHECK: .set P_SpawnMobj.uses_flat_scratch, 0
; CHECK: .set P_SpawnMobj.has_dyn_sized_stack, 0
; CHECK: .set P_SpawnMobj.has_recursion, 0
; CHECK: .set P_SpawnMobj.has_indirect_call, 0
define void @P_SpawnMobj() {
  ret void
}

; CHECK-LABEL: G_PlayerReborn:
; CHECK: .set G_PlayerReborn.num_vgpr, 0
; CHECK: .set G_PlayerReborn.num_agpr, 0
; CHECK: .set G_PlayerReborn.numbered_sgpr, 32
; CHECK: .set G_PlayerReborn.private_seg_size, 0
; CHECK: .set G_PlayerReborn.uses_vcc, 0
; CHECK: .set G_PlayerReborn.uses_flat_scratch, 0
; CHECK: .set G_PlayerReborn.has_dyn_sized_stack, 0
; CHECK: .set G_PlayerReborn.has_recursion, 0
; CHECK: .set G_PlayerReborn.has_indirect_call, 0
define void @G_PlayerReborn() {
  ret void
}

; CHECK-LABEL: P_SetThingPosition:
; CHECK: .set P_SetThingPosition.num_vgpr, 0
; CHECK: .set P_SetThingPosition.num_agpr, 0
; CHECK: .set P_SetThingPosition.numbered_sgpr, 32
; CHECK: .set P_SetThingPosition.private_seg_size, 0
; CHECK: .set P_SetThingPosition.uses_vcc, 0
; CHECK: .set P_SetThingPosition.uses_flat_scratch, 0
; CHECK: .set P_SetThingPosition.has_dyn_sized_stack, 0
; CHECK: .set P_SetThingPosition.has_recursion, 0
; CHECK: .set P_SetThingPosition.has_indirect_call, 0
define void @P_SetThingPosition() {
  ret void
}

; CHECK-LABEL: P_SetupPsprites:
; CHECK: .set P_SetupPsprites.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set P_SetupPsprites.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set P_SetupPsprites.numbered_sgpr, max(48, amdgpu.max_num_sgpr)
; CHECK: .set P_SetupPsprites.private_seg_size, 16
; CHECK: .set P_SetupPsprites.uses_vcc, 1
; CHECK: .set P_SetupPsprites.uses_flat_scratch, 1
; CHECK: .set P_SetupPsprites.has_dyn_sized_stack, 1
; CHECK: .set P_SetupPsprites.has_recursion, 1
; CHECK: .set P_SetupPsprites.has_indirect_call, 1
define void @P_SetupPsprites(ptr addrspace(1) %i) {
  %fptr = load ptr, ptr addrspace(1) %i, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: HU_Start:
; CHECK: .set HU_Start.num_vgpr, 0
; CHECK: .set HU_Start.num_agpr, 0
; CHECK: .set HU_Start.numbered_sgpr, 32
; CHECK: .set HU_Start.private_seg_size, 0
; CHECK: .set HU_Start.uses_vcc, 0
; CHECK: .set HU_Start.uses_flat_scratch, 0
; CHECK: .set HU_Start.has_dyn_sized_stack, 0
; CHECK: .set HU_Start.has_recursion, 0
; CHECK: .set HU_Start.has_indirect_call, 0
define void @HU_Start() {
  ret void
}

; CHECK-LABEL: P_SpawnPlayer:
; CHECK: .set P_SpawnPlayer.num_vgpr, max(43, G_PlayerReborn.num_vgpr, P_SetThingPosition.num_vgpr, P_SetupPsprites.num_vgpr, HU_Start.num_vgpr)
; CHECK: .set P_SpawnPlayer.num_agpr, max(0, G_PlayerReborn.num_agpr, P_SetThingPosition.num_agpr, P_SetupPsprites.num_agpr, HU_Start.num_agpr)
; CHECK: .set P_SpawnPlayer.numbered_sgpr, max(60, G_PlayerReborn.numbered_sgpr, P_SetThingPosition.numbered_sgpr, P_SetupPsprites.numbered_sgpr, HU_Start.numbered_sgpr)
; CHECK: .set P_SpawnPlayer.private_seg_size, 16+(max(G_PlayerReborn.private_seg_size, P_SetThingPosition.private_seg_size, P_SetupPsprites.private_seg_size, HU_Start.private_seg_size))
; CHECK: .set P_SpawnPlayer.uses_vcc, or(1, G_PlayerReborn.uses_vcc, P_SetThingPosition.uses_vcc, P_SetupPsprites.uses_vcc, HU_Start.uses_vcc)
; CHECK: .set P_SpawnPlayer.uses_flat_scratch, or(0, G_PlayerReborn.uses_flat_scratch, P_SetThingPosition.uses_flat_scratch, P_SetupPsprites.uses_flat_scratch, HU_Start.uses_flat_scratch)
; CHECK: .set P_SpawnPlayer.has_dyn_sized_stack, or(0, G_PlayerReborn.has_dyn_sized_stack, P_SetThingPosition.has_dyn_sized_stack, P_SetupPsprites.has_dyn_sized_stack, HU_Start.has_dyn_sized_stack)
; CHECK: .set P_SpawnPlayer.has_recursion, or(1, G_PlayerReborn.has_recursion, P_SetThingPosition.has_recursion, P_SetupPsprites.has_recursion, HU_Start.has_recursion)
; CHECK: .set P_SpawnPlayer.has_indirect_call, or(0, G_PlayerReborn.has_indirect_call, P_SetThingPosition.has_indirect_call, P_SetupPsprites.has_indirect_call, HU_Start.has_indirect_call)
define void @P_SpawnPlayer() {
  call void @G_PlayerReborn()
  call void @P_SetThingPosition()
  call void @P_SetupPsprites(ptr addrspace(1) null)
  tail call void @HU_Start()
  ret void
}

; CHECK-LABEL: I_Error:
; CHECK: .set I_Error.num_vgpr, max(41, amdgpu.max_num_vgpr)
; CHECK: .set I_Error.num_agpr, max(0, amdgpu.max_num_agpr)
; CHECK: .set I_Error.numbered_sgpr, max(48, amdgpu.max_num_sgpr)
; CHECK: .set I_Error.private_seg_size, 16
; CHECK: .set I_Error.uses_vcc, 1
; CHECK: .set I_Error.uses_flat_scratch, 1
; CHECK: .set I_Error.has_dyn_sized_stack, 1
; CHECK: .set I_Error.has_recursion, 1
; CHECK: .set I_Error.has_indirect_call, 1
define void @I_Error(...) {
  %fptr = load ptr, ptr null, align 8
  call void %fptr()
  ret void
}

; CHECK-LABEL: G_DoReborn:
; CHECK: .set G_DoReborn.num_vgpr, max(44, P_RemoveMobj.num_vgpr, P_SpawnMobj.num_vgpr, P_SpawnPlayer.num_vgpr, I_Error.num_vgpr)
; CHECK: .set G_DoReborn.num_agpr, max(0, P_RemoveMobj.num_agpr, P_SpawnMobj.num_agpr, P_SpawnPlayer.num_agpr, I_Error.num_agpr)
; CHECK: .set G_DoReborn.numbered_sgpr, max(72, P_RemoveMobj.numbered_sgpr, P_SpawnMobj.numbered_sgpr, P_SpawnPlayer.numbered_sgpr, I_Error.numbered_sgpr)
; CHECK: .set G_DoReborn.private_seg_size, 32+(max(P_RemoveMobj.private_seg_size, P_SpawnMobj.private_seg_size, P_SpawnPlayer.private_seg_size, I_Error.private_seg_size))
; CHECK: .set G_DoReborn.uses_vcc, or(1, P_RemoveMobj.uses_vcc, P_SpawnMobj.uses_vcc, P_SpawnPlayer.uses_vcc, I_Error.uses_vcc)
; CHECK: .set G_DoReborn.uses_flat_scratch, or(0, P_RemoveMobj.uses_flat_scratch, P_SpawnMobj.uses_flat_scratch, P_SpawnPlayer.uses_flat_scratch, I_Error.uses_flat_scratch)
; CHECK: .set G_DoReborn.has_dyn_sized_stack, or(0, P_RemoveMobj.has_dyn_sized_stack, P_SpawnMobj.has_dyn_sized_stack, P_SpawnPlayer.has_dyn_sized_stack, I_Error.has_dyn_sized_stack)
; CHECK: .set G_DoReborn.has_recursion, or(1, P_RemoveMobj.has_recursion, P_SpawnMobj.has_recursion, P_SpawnPlayer.has_recursion, I_Error.has_recursion)
; CHECK: .set G_DoReborn.has_indirect_call, or(0, P_RemoveMobj.has_indirect_call, P_SpawnMobj.has_indirect_call, P_SpawnPlayer.has_indirect_call, I_Error.has_indirect_call)
define void @G_DoReborn() {
  call void @P_RemoveMobj()
  call void @P_SpawnMobj()
  call void @P_SpawnPlayer()
  call void (...) @I_Error()
  ret void
}

; CHECK-LABEL: AM_Stop:
; CHECK: .set AM_Stop.num_vgpr, 0
; CHECK: .set AM_Stop.num_agpr, 0
; CHECK: .set AM_Stop.numbered_sgpr, 32
; CHECK: .set AM_Stop.private_seg_size, 0
; CHECK: .set AM_Stop.uses_vcc, 0
; CHECK: .set AM_Stop.uses_flat_scratch, 0
; CHECK: .set AM_Stop.has_dyn_sized_stack, 0
; CHECK: .set AM_Stop.has_recursion, 0
; CHECK: .set AM_Stop.has_indirect_call, 0
define void @AM_Stop() {
  ret void
}

; CHECK-LABEL: D_AdvanceDemo:
; CHECK: .set D_AdvanceDemo.num_vgpr, 0
; CHECK: .set D_AdvanceDemo.num_agpr, 0
; CHECK: .set D_AdvanceDemo.numbered_sgpr, 32
; CHECK: .set D_AdvanceDemo.private_seg_size, 0
; CHECK: .set D_AdvanceDemo.uses_vcc, 0
; CHECK: .set D_AdvanceDemo.uses_flat_scratch, 0
; CHECK: .set D_AdvanceDemo.has_dyn_sized_stack, 0
; CHECK: .set D_AdvanceDemo.has_recursion, 0
; CHECK: .set D_AdvanceDemo.has_indirect_call, 0
define void @D_AdvanceDemo() {
  ret void
}

; CHECK-LABEL: F_StartFinale:
; CHECK: .set F_StartFinale.num_vgpr, 0
; CHECK: .set F_StartFinale.num_agpr, 0
; CHECK: .set F_StartFinale.numbered_sgpr, 32
; CHECK: .set F_StartFinale.private_seg_size, 0
; CHECK: .set F_StartFinale.uses_vcc, 0
; CHECK: .set F_StartFinale.uses_flat_scratch, 0
; CHECK: .set F_StartFinale.has_dyn_sized_stack, 0
; CHECK: .set F_StartFinale.has_recursion, 0
; CHECK: .set F_StartFinale.has_indirect_call, 0
define void @F_StartFinale() {
  ret void
}

; CHECK-LABEL: F_Ticker:
; CHECK: .set F_Ticker.num_vgpr, 0
; CHECK: .set F_Ticker.num_agpr, 0
; CHECK: .set F_Ticker.numbered_sgpr, 32
; CHECK: .set F_Ticker.private_seg_size, 0
; CHECK: .set F_Ticker.uses_vcc, 0
; CHECK: .set F_Ticker.uses_flat_scratch, 0
; CHECK: .set F_Ticker.has_dyn_sized_stack, 0
; CHECK: .set F_Ticker.has_recursion, 0
; CHECK: .set F_Ticker.has_indirect_call, 0
define void @F_Ticker() {
  ret void
}

; CHECK-LABEL: G_CheckDemoStatus:
; CHECK: .set G_CheckDemoStatus.num_vgpr, max(43, I_Quit.num_vgpr, D_AdvanceDemo.num_vgpr, I_Error.num_vgpr)
; CHECK: .set G_CheckDemoStatus.num_agpr, max(0, I_Quit.num_agpr, D_AdvanceDemo.num_agpr, I_Error.num_agpr)
; CHECK: .set G_CheckDemoStatus.numbered_sgpr, max(60, I_Quit.numbered_sgpr, D_AdvanceDemo.numbered_sgpr, I_Error.numbered_sgpr)
; CHECK: .set G_CheckDemoStatus.private_seg_size, 32+(max(I_Quit.private_seg_size, D_AdvanceDemo.private_seg_size, I_Error.private_seg_size))
; CHECK: .set G_CheckDemoStatus.uses_vcc, or(1, I_Quit.uses_vcc, D_AdvanceDemo.uses_vcc, I_Error.uses_vcc)
; CHECK: .set G_CheckDemoStatus.uses_flat_scratch, or(0, I_Quit.uses_flat_scratch, D_AdvanceDemo.uses_flat_scratch, I_Error.uses_flat_scratch)
; CHECK: .set G_CheckDemoStatus.has_dyn_sized_stack, or(0, I_Quit.has_dyn_sized_stack, D_AdvanceDemo.has_dyn_sized_stack, I_Error.has_dyn_sized_stack)
; CHECK: .set G_CheckDemoStatus.has_recursion, or(1, I_Quit.has_recursion, D_AdvanceDemo.has_recursion, I_Error.has_recursion)
; CHECK: .set G_CheckDemoStatus.has_indirect_call, or(0, I_Quit.has_indirect_call, D_AdvanceDemo.has_indirect_call, I_Error.has_indirect_call)
define i32 @G_CheckDemoStatus() {
  tail call void @I_Quit()
  tail call void @D_AdvanceDemo()
  call void (...) @I_Error()
  ret i32 0
}


; CHECK-LABEL: P_TempSaveGameFile:
; CHECK: .set P_TempSaveGameFile.num_vgpr, 2
; CHECK: .set P_TempSaveGameFile.num_agpr, 0
; CHECK: .set P_TempSaveGameFile.numbered_sgpr, 32
; CHECK: .set P_TempSaveGameFile.private_seg_size, 0
; CHECK: .set P_TempSaveGameFile.uses_vcc, 0
; CHECK: .set P_TempSaveGameFile.uses_flat_scratch, 0
; CHECK: .set P_TempSaveGameFile.has_dyn_sized_stack, 0
; CHECK: .set P_TempSaveGameFile.has_recursion, 0
; CHECK: .set P_TempSaveGameFile.has_indirect_call, 0
define ptr @P_TempSaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: P_SaveGameFile:
; CHECK: .set P_SaveGameFile.num_vgpr, 2
; CHECK: .set P_SaveGameFile.num_agpr, 0
; CHECK: .set P_SaveGameFile.numbered_sgpr, 32
; CHECK: .set P_SaveGameFile.private_seg_size, 0
; CHECK: .set P_SaveGameFile.uses_vcc, 0
; CHECK: .set P_SaveGameFile.uses_flat_scratch, 0
; CHECK: .set P_SaveGameFile.has_dyn_sized_stack, 0
; CHECK: .set P_SaveGameFile.has_recursion, 0
; CHECK: .set P_SaveGameFile.has_indirect_call, 0
define ptr @P_SaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: R_FlatNumForName:
; CHECK: .set R_FlatNumForName.num_vgpr, max(42, I_Error.num_vgpr)
; CHECK: .set R_FlatNumForName.num_agpr, max(0, I_Error.num_agpr)
; CHECK: .set R_FlatNumForName.numbered_sgpr, max(48, I_Error.numbered_sgpr)
; CHECK: .set R_FlatNumForName.private_seg_size, 16+(max(I_Error.private_seg_size))
; CHECK: .set R_FlatNumForName.uses_vcc, or(1, I_Error.uses_vcc)
; CHECK: .set R_FlatNumForName.uses_flat_scratch, or(0, I_Error.uses_flat_scratch)
; CHECK: .set R_FlatNumForName.has_dyn_sized_stack, or(0, I_Error.has_dyn_sized_stack)
; CHECK: .set R_FlatNumForName.has_recursion, or(1, I_Error.has_recursion)
; CHECK: .set R_FlatNumForName.has_indirect_call, or(0, I_Error.has_indirect_call)
define i32 @R_FlatNumForName() {
  call void (...) @I_Error()
  unreachable
}

; CHECK-LABEL: R_TextureNumForName:
; CHECK: .set R_TextureNumForName.num_vgpr, max(42, R_FlatNumForName.num_vgpr)
; CHECK: .set R_TextureNumForName.num_agpr, max(0, R_FlatNumForName.num_agpr)
; CHECK: .set R_TextureNumForName.numbered_sgpr, max(48, R_FlatNumForName.numbered_sgpr)
; CHECK: .set R_TextureNumForName.private_seg_size, 16+(max(R_FlatNumForName.private_seg_size))
; CHECK: .set R_TextureNumForName.uses_vcc, or(1, R_FlatNumForName.uses_vcc)
; CHECK: .set R_TextureNumForName.uses_flat_scratch, or(0, R_FlatNumForName.uses_flat_scratch)
; CHECK: .set R_TextureNumForName.has_dyn_sized_stack, or(0, R_FlatNumForName.has_dyn_sized_stack)
; CHECK: .set R_TextureNumForName.has_recursion, or(1, R_FlatNumForName.has_recursion)
; CHECK: .set R_TextureNumForName.has_indirect_call, or(0, R_FlatNumForName.has_indirect_call)
define i32 @R_TextureNumForName() {
  %ret = call i32 @R_FlatNumForName()
  ret i32 0
}

; CHECK-LABEL: G_Ticker:
; CHECK: .set G_Ticker.num_vgpr, max(46, G_DoReborn.num_vgpr, F_Ticker.num_vgpr, AM_Stop.num_vgpr, F_StartFinale.num_vgpr, D_AdvanceDemo.num_vgpr, R_FlatNumForName.num_vgpr, R_TextureNumForName.num_vgpr, P_TempSaveGameFile.num_vgpr, P_SaveGameFile.num_vgpr, I_Error.num_vgpr)
; CHECK: .set G_Ticker.num_agpr, max(0, G_DoReborn.num_agpr, F_Ticker.num_agpr, AM_Stop.num_agpr, F_StartFinale.num_agpr, D_AdvanceDemo.num_agpr, R_FlatNumForName.num_agpr, R_TextureNumForName.num_agpr, P_TempSaveGameFile.num_agpr, P_SaveGameFile.num_agpr, I_Error.num_agpr)
; CHECK: .set G_Ticker.numbered_sgpr, max(84, G_DoReborn.numbered_sgpr, F_Ticker.numbered_sgpr, AM_Stop.numbered_sgpr, F_StartFinale.numbered_sgpr, D_AdvanceDemo.numbered_sgpr, R_FlatNumForName.numbered_sgpr, R_TextureNumForName.numbered_sgpr, P_TempSaveGameFile.numbered_sgpr, P_SaveGameFile.numbered_sgpr, I_Error.numbered_sgpr)
; CHECK: .set G_Ticker.private_seg_size, 32+(max(G_DoReborn.private_seg_size, F_Ticker.private_seg_size, AM_Stop.private_seg_size, F_StartFinale.private_seg_size, D_AdvanceDemo.private_seg_size, R_FlatNumForName.private_seg_size, R_TextureNumForName.private_seg_size, P_TempSaveGameFile.private_seg_size, P_SaveGameFile.private_seg_size, I_Error.private_seg_size))
; CHECK: .set G_Ticker.uses_vcc, or(1, G_DoReborn.uses_vcc, F_Ticker.uses_vcc, AM_Stop.uses_vcc, F_StartFinale.uses_vcc, D_AdvanceDemo.uses_vcc, R_FlatNumForName.uses_vcc, R_TextureNumForName.uses_vcc, P_TempSaveGameFile.uses_vcc, P_SaveGameFile.uses_vcc, I_Error.uses_vcc)
; CHECK: .set G_Ticker.uses_flat_scratch, or(0, G_DoReborn.uses_flat_scratch, F_Ticker.uses_flat_scratch, AM_Stop.uses_flat_scratch, F_StartFinale.uses_flat_scratch, D_AdvanceDemo.uses_flat_scratch, R_FlatNumForName.uses_flat_scratch, R_TextureNumForName.uses_flat_scratch, P_TempSaveGameFile.uses_flat_scratch, P_SaveGameFile.uses_flat_scratch, I_Error.uses_flat_scratch)
; CHECK: .set G_Ticker.has_dyn_sized_stack, or(0, G_DoReborn.has_dyn_sized_stack, F_Ticker.has_dyn_sized_stack, AM_Stop.has_dyn_sized_stack, F_StartFinale.has_dyn_sized_stack, D_AdvanceDemo.has_dyn_sized_stack, R_FlatNumForName.has_dyn_sized_stack, R_TextureNumForName.has_dyn_sized_stack, P_TempSaveGameFile.has_dyn_sized_stack, P_SaveGameFile.has_dyn_sized_stack, I_Error.has_dyn_sized_stack)
; CHECK: .set G_Ticker.has_recursion, or(1, G_DoReborn.has_recursion, F_Ticker.has_recursion, AM_Stop.has_recursion, F_StartFinale.has_recursion, D_AdvanceDemo.has_recursion, R_FlatNumForName.has_recursion, R_TextureNumForName.has_recursion, P_TempSaveGameFile.has_recursion, P_SaveGameFile.has_recursion, I_Error.has_recursion)
; CHECK: .set G_Ticker.has_indirect_call, or(0, G_DoReborn.has_indirect_call, F_Ticker.has_indirect_call, AM_Stop.has_indirect_call, F_StartFinale.has_indirect_call, D_AdvanceDemo.has_indirect_call, R_FlatNumForName.has_indirect_call, R_TextureNumForName.has_indirect_call, P_TempSaveGameFile.has_indirect_call, P_SaveGameFile.has_indirect_call, I_Error.has_indirect_call)
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
; CHECK: .set RunTic.num_vgpr, max(46, G_CheckDemoStatus.num_vgpr, D_AdvanceDemo.num_vgpr, G_Ticker.num_vgpr)
; CHECK: .set RunTic.num_agpr, max(0, G_CheckDemoStatus.num_agpr, D_AdvanceDemo.num_agpr, G_Ticker.num_agpr)
; CHECK: .set RunTic.numbered_sgpr, max(84, G_CheckDemoStatus.numbered_sgpr, D_AdvanceDemo.numbered_sgpr, G_Ticker.numbered_sgpr)
; CHECK: .set RunTic.private_seg_size, 32+(max(G_CheckDemoStatus.private_seg_size, D_AdvanceDemo.private_seg_size, G_Ticker.private_seg_size))
; CHECK: .set RunTic.uses_vcc, or(1, G_CheckDemoStatus.uses_vcc, D_AdvanceDemo.uses_vcc, G_Ticker.uses_vcc)
; CHECK: .set RunTic.uses_flat_scratch, or(0, G_CheckDemoStatus.uses_flat_scratch, D_AdvanceDemo.uses_flat_scratch, G_Ticker.uses_flat_scratch)
; CHECK: .set RunTic.has_dyn_sized_stack, or(0, G_CheckDemoStatus.has_dyn_sized_stack, D_AdvanceDemo.has_dyn_sized_stack, G_Ticker.has_dyn_sized_stack)
; CHECK: .set RunTic.has_recursion, or(1, G_CheckDemoStatus.has_recursion, D_AdvanceDemo.has_recursion, G_Ticker.has_recursion)
; CHECK: .set RunTic.has_indirect_call, or(0, G_CheckDemoStatus.has_indirect_call, D_AdvanceDemo.has_indirect_call, G_Ticker.has_indirect_call)
define void @RunTic() {
  %call5.i1 = call i32 @G_CheckDemoStatus()
  tail call void @D_AdvanceDemo()
  call void @G_Ticker()
  ret void
}

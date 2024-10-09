; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s

; Previously, this would hit an assertion on incompatible comparison between
; APInts due to BitWidth differences. This was due to assignment of DenseMap
; value using another value within that same DenseMap which results in a
; use-after-free if the assignment operator invokes a DenseMap growth.

; CHECK-LABEL: I_Quit:
define void @I_Quit() {
  %fptr = load ptr, ptr null, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: P_RemoveMobj:
define void @P_RemoveMobj() {
  ret void
}

; CHECK-LABEL: P_SpawnMobj:
define void @P_SpawnMobj() {
  ret void
}

; CHECK-LABEL: G_PlayerReborn:
define void @G_PlayerReborn() {
  ret void
}

; CHECK-LABEL: P_SetThingPosition:
define void @P_SetThingPosition() {
  ret void
}

; CHECK-LABEL: P_SetupPsprites:
define void @P_SetupPsprites(ptr addrspace(1) %i) {
  %fptr = load ptr, ptr addrspace(1) %i, align 8
  tail call void %fptr()
  ret void
}

; CHECK-LABEL: HU_Start:
define void @HU_Start() {
  ret void
}

; CHECK-LABEL: P_SpawnPlayer:
define void @P_SpawnPlayer() {
  call void @G_PlayerReborn()
  call void @P_SetThingPosition()
  call void @P_SetupPsprites(ptr addrspace(1) null)
  tail call void @HU_Start()
  ret void
}

; CHECK-LABEL: I_Error:
define void @I_Error(...) {
  %fptr = load ptr, ptr null, align 8
  call void %fptr()
  ret void
}

; CHECK-LABEL: G_DoReborn:
define void @G_DoReborn() {
  call void @P_RemoveMobj()
  call void @P_SpawnMobj()
  call void @P_SpawnPlayer()
  call void (...) @I_Error()
  ret void
}

; CHECK-LABEL: AM_Stop:
define void @AM_Stop() {
  ret void
}

; CHECK-LABEL: D_AdvanceDemo:
define void @D_AdvanceDemo() {
  ret void
}

; CHECK-LABEL: F_StartFinale:
define void @F_StartFinale() {
  ret void
}

; CHECK-LABEL: F_Ticker:
define void @F_Ticker() {
  ret void
}

; CHECK-LABEL: G_CheckDemoStatus:
define i32 @G_CheckDemoStatus() {
  tail call void @I_Quit()
  tail call void @D_AdvanceDemo()
  call void (...) @I_Error()
  ret i32 0
}


; CHECK-LABEL: P_TempSaveGameFile:
define ptr @P_TempSaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: P_SaveGameFile:
define ptr @P_SaveGameFile() {
  ret ptr null
}

; CHECK-LABEL: R_FlatNumForName:
define i32 @R_FlatNumForName() {
  call void (...) @I_Error()
  unreachable
}

; CHECK-LABEL: R_TextureNumForName:
define i32 @R_TextureNumForName() {
  %ret = call i32 @R_FlatNumForName()
  ret i32 0
}

; CHECK-LABEL: G_Ticker:
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
define void @RunTic() {
  %call5.i1 = call i32 @G_CheckDemoStatus()
  tail call void @D_AdvanceDemo()
  call void @G_Ticker()
  ret void
}

; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -o - %s 2>&1 | FileCheck %s

; CHECK-NOT: Assertion `BitWidth == RHS.BitWidth && "Bit widths must be same for comparison"' failed

define void @RunTic() {
  %call5.i1 = call i32 @G_CheckDemoStatus()
  tail call void @D_AdvanceDemo()
  call void @G_Ticker()
  ret void
}

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

define void @G_DoReborn() {
  call void @P_RemoveMobj()
  call void @P_SpawnMobj()
  call void @P_SpawnPlayer()
  call void (...) @I_Error()
  ret void
}

define void @AM_Stop() {
  ret void
}

define void @D_AdvanceDemo() {
  ret void
}

define void @F_StartFinale() {
  ret void
}

define void @F_Ticker() {
  ret void
}

define void @G_PlayerReborn() {
  ret void
}

define i32 @G_CheckDemoStatus() {
  tail call void @I_Quit()
  tail call void @D_AdvanceDemo()
  call void (...) @I_Error()
  ret i32 0
}

define void @HU_Start() {
  ret void
}

define void @I_Quit() {
  %fptr = load ptr, ptr null, align 8
  tail call void %fptr()
  ret void
}

define void @P_SetThingPosition() {
  ret void
}

define void @P_RemoveMobj() {
  ret void
}

define void @P_SpawnMobj() {
  ret void
}

define void @P_SpawnPlayer() {
  call void @G_PlayerReborn()
  call void @P_SetThingPosition()
  call void @P_SetupPsprites(ptr addrspace(1) null)
  tail call void @HU_Start()
  ret void
}

define void @P_SetupPsprites(ptr addrspace(1) %i) {
  %fptr = load ptr, ptr addrspace(1) %i, align 8
  tail call void %fptr()
  ret void
}

define ptr @P_TempSaveGameFile() {
  ret ptr null
}

define ptr @P_SaveGameFile() {
  ret ptr null
}

define i32 @R_TextureNumForName() {
  %ret = call i32 @R_FlatNumForName()
  ret i32 0
}

define void @I_Error(...) {
  %fptr = load ptr, ptr null, align 8
  call void %fptr()
  ret void
}

define i32 @R_FlatNumForName() {
  call void (...) @I_Error()
  unreachable
}

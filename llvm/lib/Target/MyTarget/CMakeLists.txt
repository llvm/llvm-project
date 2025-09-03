# 告訴 tblgen 從哪個 .td 當入口
set(LLVM_TARGET_DEFINITIONS MyTarget.td)

# 產生你需要的 .inc（至少有 InstrInfo；你有 RegisterClass，順便生 RegisterInfo）
tablegen(LLVM MyTargetGenInstrInfo.inc    -gen-instr-info)
tablegen(LLVM MyTargetGenRegisterInfo.inc -gen-register-info)

# 暴露 tablegen 目標給後面的 target 用
add_public_tablegen_target(MyTargetCommonTableGen)

# 編譯這個後端的 C++ 檔，並確保 .inc 已經生好
add_llvm_target(MyTargetCodeGen
  MyTargetInstrInfo.cpp

  DEPENDS
    MyTargetCommonTableGen
)

# ★ 關鍵：建立「傘狀」目標，名稱必須就是 MyTarget
# 頂層會查詢這個名字；用它來依賴到真正的 CodeGen 目標
add_llvm_component_group(MyTarget)
add_dependencies(MyTarget LLVMMyTargetCodeGen)

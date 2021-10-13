_T = require('lua_lldb_test').create_test('TestComprehensive')

function _T:Test0_CreateTarget()
    self.target = self:create_target()
    assertTrue(self.target:IsValid())
end

function _T:Test1_Breakpoint()
    self.main_bp = self.target:BreakpointCreateByName('main', 'a.out')
    self.loop_bp = self.target:BreakpointCreateByLocation('main.c', 28)
    assertTrue(self.main_bp:IsValid() and self.main_bp:GetNumLocations() == 1)
    assertTrue(self.loop_bp:IsValid() and self.loop_bp:GetNumLocations() == 1)
end

function _T:Test2_Launch()
    local error = lldb.SBError()
    self.args = { 'arg' }
    self.process = self.target:Launch(
        self.debugger:GetListener(),
        self.args,
        nil,
        nil,
        self.output,
        nil,
        nil,
        0,
        false,
        error
    )
    assertTrue(error:Success())
    assertTrue(self.process:IsValid())
end

function _T:Test3_BreakpointFindVariables()
    -- checking "argc" value
    local thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
    assertNotNil(thread)
    assertTrue(thread:IsValid())
    local frame = thread:GetFrameAtIndex(0)
    assertTrue(frame:IsValid())
    local error = lldb.SBError()
    local var_argc = frame:FindVariable('argc')
    assertTrue(var_argc:IsValid())
    local var_argc_value = var_argc:GetValueAsSigned(error, 0)
    assertTrue(error:Success())
    assertEquals(var_argc_value, 2)

    -- checking "inited" value
    local continue = self.process:Continue()
    assertTrue(continue:Success())
    thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
    assertNotNil(thread)
    assertTrue(thread:IsValid())
    frame = thread:GetFrameAtIndex(0)
    assertTrue(frame:IsValid())
    error = lldb.SBError()
    local var_inited = frame:FindVariable('inited')
    assertTrue(var_inited:IsValid())
    self.var_inited = var_inited
    local var_inited_value = var_inited:GetValueAsUnsigned(error, 0)
    assertTrue(error:Success())
    assertEquals(var_inited_value, 0xDEADBEEF)
end

function _T:Test3_RawData()
    local error = lldb.SBError()
    local address = self.var_inited:GetAddress()
    assertTrue(address:IsValid())
    local size = self.var_inited:GetByteSize()
    local raw_data = self.process:ReadMemory(address:GetOffset(), size, error)
    assertTrue(error:Success())
    local data_le = lldb.SBData.CreateDataFromUInt32Array(lldb.eByteOrderLittle, 1, {0xDEADBEEF})
    local data_be = lldb.SBData.CreateDataFromUInt32Array(lldb.eByteOrderBig, 1, {0xDEADBEEF})
    assertTrue(data_le:GetUnsignedInt32(error, 0) == 0xDEADBEEF or data_be:GetUnsignedInt32(error, 0) == 0xDEADBEEF)
    assertTrue(raw_data == "\xEF\xBE\xAD\xDE" or raw_data == "\xDE\xAD\xBE\xEF")
end

function _T:Test4_ProcessExit()
    self.loop_bp:SetAutoContinue(true)
    local continue = self.process:Continue()
    assertTrue(continue:Success())
    assertTrue(self.process:GetExitStatus() == 0)
end

function _T:Test5_FileOutput()
    local f = io.open(self.output, 'r')
    assertEquals(
        read_file_non_empty_lines(f),
        {
            self.exe,
            table.unpack(self.args),
            'I am a function.',
            'sum = 5050'
        }
    )
    f:close()
end

os.exit(_T:run())

_T = require('lua_lldb_test').create_test('TestProcessAPI')

function _T:TestProcessLaunchSimple()
    local target = self:create_target()
    local args = { 'arg1', 'arg2', 'arg3' }
    local process = target:LaunchSimple(
        -- argv
        args,
        -- envp
        nil,
        -- working directory
        nil
    )
    assertTrue(process:IsValid())
    local stdout = process:GetSTDOUT(1000)
    assertEqual(split_lines(stdout), {self.exe, table.unpack(args)})
end

function _T:TestProcessLaunch()
    local target = self:create_target()
    local args = { 'arg1', 'arg2', 'arg3' }
    local error = lldb.SBError()
    local f = io.open(self.output, 'w')
    f:write()
    f:close()
    local process = target:Launch(
        -- listener
        self.debugger:GetListener(),
        -- argv
        args,
        -- envp
        nil,
        -- stdin
        nil,
        -- stdout
        self.output,
        -- stderr
        nil,
        -- working directory
        nil,
        -- launch flags
        0,
        -- stop at entry
        true,
        -- error
        error
    )
    assertTrue(error:Success())
    assertTrue(process:IsValid())
    local threads = get_stopped_threads(process, lldb.eStopReasonSignal)
    assertTrue(#threads ~= 0)
    local continue = process:Continue()
    assertTrue(continue:Success())
    local f = io.open(self.output, 'r')
    assertEqual(read_file_non_empty_lines(f), {self.exe, table.unpack(args)})
    f:close()
end

os.exit(_T:run())

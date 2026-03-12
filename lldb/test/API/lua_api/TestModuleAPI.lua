_T = require('lua_lldb_test').create_test('TestModuleAPI')

function _T:TestGetAndSetUUID()
    local target = self:create_target()
    local module = target:GetModuleAtIndex(0)
    assertTrue(module:IsValid())

    local module_uuid_bytes = module:GetUUIDBytes()
    assertTrue(type(module_uuid_bytes) == "string")
    local module_uuid_string = module:GetUUIDString()
    assertTrue(type(module_uuid_string) == "string")
    local expected_bytes_uuid = module_uuid_bytes:gsub(".",
        function(c) return string.format("%02x", string.byte(c)) end):upper()
    assertEqual(module_uuid_string:gsub("-", ""), expected_bytes_uuid)

    local spec = lldb.SBModuleSpec()
    local bytesres_type = type(spec:GetUUIDBytes())
    assertTrue(bytesres_type == "nil")
    local spec_bytes_uuid = "8FB5E28E344ECA77CE1969FD79A9B72AFD27C88F"

    assertTrue(spec:SetUUIDBytes(spec_bytes_uuid))
    assertEqual(spec:GetUUIDBytes(), spec_bytes_uuid:upper())
end
os.exit(_T:run())

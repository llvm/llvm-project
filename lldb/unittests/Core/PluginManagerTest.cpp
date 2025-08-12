
#include "lldb/Core/PluginManager.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

// Mock system runtime plugin create functions.
// Make them all return different values to avoid the ICF optimization
// from combining them into the same function. The values returned
// are not valid SystemRuntime pointers, but they are unique and
// sufficient for testing.
SystemRuntime *CreateSystemRuntimePluginA(Process *process) {
  return (SystemRuntime *)0x1;
}

SystemRuntime *CreateSystemRuntimePluginB(Process *process) {
  return (SystemRuntime *)0x2;
}

SystemRuntime *CreateSystemRuntimePluginC(Process *process) {
  return (SystemRuntime *)0x3;
}

// Test class for testing the PluginManager.
// The PluginManager modifies global state when registering new plugins. This
// class is intended to undo those modifications in the destructor to give each
// test a clean slate with no registered plugins at the start of a test.
class PluginManagerTest : public testing::Test {
public:
  // Remove any pre-registered plugins so we have a known starting point.
  static void SetUpTestSuite() { RemoveAllRegisteredSystemRuntimePlugins(); }

  // Add mock system runtime plugins for testing.
  void RegisterMockSystemRuntimePlugins() {
    // Make sure the create functions all have different addresses.
    ASSERT_NE(CreateSystemRuntimePluginA, CreateSystemRuntimePluginB);
    ASSERT_NE(CreateSystemRuntimePluginB, CreateSystemRuntimePluginC);

    ASSERT_TRUE(PluginManager::RegisterPlugin("a", "test instance A",
                                              CreateSystemRuntimePluginA));
    ASSERT_TRUE(PluginManager::RegisterPlugin("b", "test instance B",
                                              CreateSystemRuntimePluginB));
    ASSERT_TRUE(PluginManager::RegisterPlugin("c", "test instance C",
                                              CreateSystemRuntimePluginC));
  }

  // Remove any plugins added during the tests.
  virtual ~PluginManagerTest() override {
    RemoveAllRegisteredSystemRuntimePlugins();
  }

protected:
  std::vector<SystemRuntimeCreateInstance> m_system_runtime_plugins;

  static void RemoveAllRegisteredSystemRuntimePlugins() {
    // Enable all currently registered plugins so we can get a handle to
    // their create callbacks in the loop below. Only enabled plugins
    // are returned from the PluginManager Get*CreateCallbackAtIndex apis.
    for (const RegisteredPluginInfo &PluginInfo :
         PluginManager::GetSystemRuntimePluginInfo()) {
      PluginManager::SetSystemRuntimePluginEnabled(PluginInfo.name, true);
    }

    // Get a handle to the create call backs for all the registered plugins.
    std::vector<SystemRuntimeCreateInstance> registered_plugin_callbacks;
    SystemRuntimeCreateInstance create_callback = nullptr;
    for (uint32_t idx = 0;
         (create_callback =
              PluginManager::GetSystemRuntimeCreateCallbackAtIndex(idx)) !=
         nullptr;
         ++idx) {
      registered_plugin_callbacks.push_back((create_callback));
    }

    // Remove all currently registered plugins.
    for (SystemRuntimeCreateInstance create_callback :
         registered_plugin_callbacks) {
      PluginManager::UnregisterPlugin(create_callback);
    }
  }
};

// Test basic register functionality.
TEST_F(PluginManagerTest, RegisterSystemRuntimePlugin) {
  RegisterMockSystemRuntimePlugins();

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginC);

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(3), nullptr);
}

// Test basic un-register functionality.
TEST_F(PluginManagerTest, UnRegisterSystemRuntimePlugin) {
  RegisterMockSystemRuntimePlugins();

  ASSERT_TRUE(PluginManager::UnregisterPlugin(CreateSystemRuntimePluginB));

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);

  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2), nullptr);
}

// Test registered plugin info functionality.
TEST_F(PluginManagerTest, SystemRuntimePluginInfo) {
  RegisterMockSystemRuntimePlugins();

  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].description, "test instance A");
  ASSERT_EQ(plugin_info[0].enabled, true);
  ASSERT_EQ(plugin_info[1].name, "b");
  ASSERT_EQ(plugin_info[1].description, "test instance B");
  ASSERT_EQ(plugin_info[1].enabled, true);
  ASSERT_EQ(plugin_info[2].name, "c");
  ASSERT_EQ(plugin_info[2].description, "test instance C");
  ASSERT_EQ(plugin_info[2].enabled, true);
}

// Test basic un-register functionality.
TEST_F(PluginManagerTest, UnRegisterSystemRuntimePluginInfo) {
  RegisterMockSystemRuntimePlugins();

  // Initial plugin info has all three registered plugins.
  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);

  ASSERT_TRUE(PluginManager::UnregisterPlugin(CreateSystemRuntimePluginB));

  // After un-registering a plugin it should be removed from plugin info.
  plugin_info = PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 2u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].enabled, true);
  ASSERT_EQ(plugin_info[1].name, "c");
  ASSERT_EQ(plugin_info[1].enabled, true);
}

// Test plugin disable functionality.
TEST_F(PluginManagerTest, SystemRuntimePluginDisable) {
  RegisterMockSystemRuntimePlugins();

  // Disable plugin should succeed.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));

  // Disabling a plugin does not remove it from plugin info.
  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].enabled, true);
  ASSERT_EQ(plugin_info[1].name, "b");
  ASSERT_EQ(plugin_info[1].enabled, false);
  ASSERT_EQ(plugin_info[2].name, "c");
  ASSERT_EQ(plugin_info[2].enabled, true);

  // Disabling a plugin does remove it from available plugins.
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2), nullptr);
}

// Test plugin disable and enable functionality.
TEST_F(PluginManagerTest, SystemRuntimePluginDisableThenEnable) {
  RegisterMockSystemRuntimePlugins();

  // Initially plugin b is available in slot 1.
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);

  // Disabling it will remove it from available plugins.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);

  // We can re-enable the plugin later and it should go back to the original
  // slot.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", true));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginC);

  // And show up in the plugin info correctly.
  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].enabled, true);
  ASSERT_EQ(plugin_info[1].name, "b");
  ASSERT_EQ(plugin_info[1].enabled, true);
  ASSERT_EQ(plugin_info[2].name, "c");
  ASSERT_EQ(plugin_info[2].enabled, true);
}

// Test calling disable on an already disabled plugin is ok.
TEST_F(PluginManagerTest, SystemRuntimePluginDisableDisabled) {
  RegisterMockSystemRuntimePlugins();

  // Initial call to disable the plugin should succeed.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));

  // The second call should also succeed because the plugin is already disabled.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));

  // The call to re-enable the plugin should succeed.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", true));

  // The second call should also succeed since the plugin is already enabled.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", true));
}

// Test calling disable on an already disabled plugin is ok.
TEST_F(PluginManagerTest, SystemRuntimePluginDisableNonExistent) {
  RegisterMockSystemRuntimePlugins();

  // Both enable and disable should return false for a non-existent plugin.
  ASSERT_FALSE(
      PluginManager::SetSystemRuntimePluginEnabled("does_not_exist", true));
  ASSERT_FALSE(
      PluginManager::SetSystemRuntimePluginEnabled("does_not_exist", false));
}

// Test disabling all plugins and then re-enabling them in a different
// order will restore the original plugin order.
TEST_F(PluginManagerTest, SystemRuntimePluginDisableAll) {
  RegisterMockSystemRuntimePlugins();

  // Validate initial state of registered plugins.
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginC);

  // Disable all the active plugins.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("a", false));
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("c", false));

  // Should have no active plugins.
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0), nullptr);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1), nullptr);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2), nullptr);

  // And show up in the plugin info correctly.
  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].enabled, false);
  ASSERT_EQ(plugin_info[1].name, "b");
  ASSERT_EQ(plugin_info[1].enabled, false);
  ASSERT_EQ(plugin_info[2].name, "c");
  ASSERT_EQ(plugin_info[2].enabled, false);

  // Enable plugins in reverse order and validate expected indicies.
  // They should show up in the original plugin order.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("c", true));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginC);

  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("a", true));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);

  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", true));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginC);
}

// Test un-registering a disabled plugin works.
TEST_F(PluginManagerTest, UnRegisterDisabledSystemRuntimePlugin) {
  RegisterMockSystemRuntimePlugins();

  // Initial plugin info has all three registered plugins.
  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 3u);

  // First disable a plugin, then unregister it. Both should succeed.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));
  ASSERT_TRUE(PluginManager::UnregisterPlugin(CreateSystemRuntimePluginB));

  // After un-registering a plugin it should be removed from plugin info.
  plugin_info = PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(plugin_info.size(), 2u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[0].enabled, true);
  ASSERT_EQ(plugin_info[1].name, "c");
  ASSERT_EQ(plugin_info[1].enabled, true);
}

// Test un-registering and then re-registering a plugin will change the order of
// loaded plugins.
TEST_F(PluginManagerTest, UnRegisterSystemRuntimePluginChangesOrder) {
  RegisterMockSystemRuntimePlugins();

  std::vector<RegisteredPluginInfo> plugin_info =
      PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginC);

  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[1].name, "b");
  ASSERT_EQ(plugin_info[2].name, "c");

  // Unregister and then registering a plugin puts it at the end of the order
  // list.
  ASSERT_TRUE(PluginManager::UnregisterPlugin(CreateSystemRuntimePluginB));
  ASSERT_TRUE(PluginManager::RegisterPlugin("b", "New test instance B",
                                            CreateSystemRuntimePluginB));

  // Check the callback indices match as expected.
  plugin_info = PluginManager::GetSystemRuntimePluginInfo();
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginB);

  // And plugin info should match as well.
  ASSERT_EQ(plugin_info.size(), 3u);
  ASSERT_EQ(plugin_info[0].name, "a");
  ASSERT_EQ(plugin_info[1].name, "c");
  ASSERT_EQ(plugin_info[2].name, "b");
  ASSERT_EQ(plugin_info[2].description, "New test instance B");

  // Disabling and re-enabling the "c" plugin should slot it back
  // into the middle of the order. Originally it was last, but after
  // un-registering and re-registering "b" it should now stay in
  // the middle of the order.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("c", false));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginB);

  // And re-enabling
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("c", true));
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(0),
            CreateSystemRuntimePluginA);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(1),
            CreateSystemRuntimePluginC);
  ASSERT_EQ(PluginManager::GetSystemRuntimeCreateCallbackAtIndex(2),
            CreateSystemRuntimePluginB);
}

TEST_F(PluginManagerTest, MatchPluginName) {
  PluginNamespace Foo{"foo", nullptr, nullptr};
  RegisteredPluginInfo Bar{"bar", "bar plugin ", true};
  RegisteredPluginInfo Baz{"baz", "baz plugin ", true};

  // Empty pattern matches everything.
  ASSERT_TRUE(PluginManager::MatchPluginName("", Foo, Bar));

  // Plugin namespace matches all plugins in that namespace.
  ASSERT_TRUE(PluginManager::MatchPluginName("foo", Foo, Bar));
  ASSERT_TRUE(PluginManager::MatchPluginName("foo", Foo, Baz));

  // Fully qualified plugin name matches only that plugin.
  ASSERT_TRUE(PluginManager::MatchPluginName("foo.bar", Foo, Bar));
  ASSERT_FALSE(PluginManager::MatchPluginName("foo.baz", Foo, Bar));

  // Prefix match should not match.
  ASSERT_FALSE(PluginManager::MatchPluginName("f", Foo, Bar));
  ASSERT_FALSE(PluginManager::MatchPluginName("foo.", Foo, Bar));
  ASSERT_FALSE(PluginManager::MatchPluginName("foo.ba", Foo, Bar));
}

TEST_F(PluginManagerTest, JsonFormat) {
  RegisterMockSystemRuntimePlugins();

  // We expect the following JSON output:
  // {
  //   "system-runtime": [
  //     {
  //       "enabled": true,
  //       "name": "a"
  //     },
  //     {
  //       "enabled": true,
  //       "name": "b"
  //     },
  //     {
  //       "enabled": true,
  //       "name": "c"
  //     }
  //   ]
  // }
  llvm::json::Object obj = PluginManager::GetJSON();

  // We should have a "system-runtime" array in the top-level object.
  llvm::json::Array *maybe_array = obj.getArray("system-runtime");
  ASSERT_TRUE(maybe_array != nullptr);
  auto &array = *maybe_array;
  ASSERT_EQ(array.size(), 3u);

  // Check plugin "a" info.
  ASSERT_TRUE(array[0].getAsObject() != nullptr);
  ASSERT_TRUE(array[0].getAsObject()->getString("name") == "a");
  ASSERT_TRUE(array[0].getAsObject()->getBoolean("enabled") == true);

  // Check plugin "b" info.
  ASSERT_TRUE(array[1].getAsObject() != nullptr);
  ASSERT_TRUE(array[1].getAsObject()->getString("name") == "b");
  ASSERT_TRUE(array[1].getAsObject()->getBoolean("enabled") == true);

  // Check plugin "c" info.
  ASSERT_TRUE(array[2].getAsObject() != nullptr);
  ASSERT_TRUE(array[2].getAsObject()->getString("name") == "c");
  ASSERT_TRUE(array[2].getAsObject()->getBoolean("enabled") == true);

  // Disabling a plugin should be reflected in the JSON output.
  ASSERT_TRUE(PluginManager::SetSystemRuntimePluginEnabled("b", false));
  array = *PluginManager::GetJSON().getArray("system-runtime");
  ASSERT_TRUE(array[0].getAsObject()->getBoolean("enabled") == true);
  ASSERT_TRUE(array[1].getAsObject()->getBoolean("enabled") == false);
  ASSERT_TRUE(array[2].getAsObject()->getBoolean("enabled") == true);

  // Un-registering a plugin should be reflected in the JSON output.
  ASSERT_TRUE(PluginManager::UnregisterPlugin(CreateSystemRuntimePluginB));
  array = *PluginManager::GetJSON().getArray("system-runtime");
  ASSERT_EQ(array.size(), 2u);
  ASSERT_TRUE(array[0].getAsObject()->getString("name") == "a");
  ASSERT_TRUE(array[1].getAsObject()->getString("name") == "c");

  // Filtering the JSON output should only include the matching plugins.
  array =
      *PluginManager::GetJSON("system-runtime.c").getArray("system-runtime");
  ASSERT_EQ(array.size(), 1u);
  ASSERT_TRUE(array[0].getAsObject()->getString("name") == "c");

  // Empty JSON output is allowed if there are no matching plugins.
  obj = PluginManager::GetJSON("non-existent-plugin");
  ASSERT_TRUE(obj.empty());
}

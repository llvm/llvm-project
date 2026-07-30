import * as assert from "assert";

import {
  applyEnvironmentOverrides,
  setEnvironmentValue,
} from "../../src/compatibility-utils";

suite("compatibility-utils environment helpers", function () {
  test("setEnvironmentValue reuses the existing key's casing", function () {
    const env = { Path: "C:\\Windows" };

    setEnvironmentValue(env, "PATH", "C:\\NewDir");

    assert.deepStrictEqual(env, { Path: "C:\\NewDir" });
  });

  test("setEnvironmentValue creates the key as given when absent", function () {
    const env: { [key: string]: string } = {};

    setEnvironmentValue(env, "FOO", "bar");

    assert.deepStrictEqual(env, { FOO: "bar" });
  });

  test("applyEnvironmentOverrides merges case-insensitively on win32", function () {
    const target = { Path: "C:\\Windows" };

    applyEnvironmentOverrides(target, { PATH: "C:\\Override" }, "win32");

    // Only one key should remain, since "PATH" and "Path" refer to the same
    // environment variable on Windows.
    assert.deepStrictEqual(target, { Path: "C:\\Override" });
  });

  test("applyEnvironmentOverrides keeps distinct keys on non-Windows platforms", function () {
    const target = { Path: "C:\\Windows" };

    applyEnvironmentOverrides(target, { PATH: "/usr/bin" }, "linux");

    assert.deepStrictEqual(target, { Path: "C:\\Windows", PATH: "/usr/bin" });
  });
});

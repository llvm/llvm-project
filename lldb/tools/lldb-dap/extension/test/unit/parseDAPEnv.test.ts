import * as assert from "assert";
import { parseDAPEnv } from "../../src/utils";

suite("parseDAPEnv", function () {
  test("parses a dictionary", function () {
    const value = { FOO: "1", _BAR: "", BAZ: "1\n2" };
    assert.deepStrictEqual(parseDAPEnv(value), value);

    const emptyValue = {};
    assert.deepStrictEqual(parseDAPEnv(emptyValue), emptyValue);
  });

  test("parses an array", function () {
    const actual = ["FOO=a=b", "_BAR", "BAZ="];
    const expected = { FOO: "a=b", _BAR: "", BAZ: "" };
    assert.deepStrictEqual(parseDAPEnv(actual), expected);

    assert.deepStrictEqual(parseDAPEnv([]), {});
  });

  test("rejects keys starting with a digit", function () {
    assert.strictEqual(parseDAPEnv({ "1FOO": "1" }), undefined);
    assert.strictEqual(parseDAPEnv(["1FOO=1"]), undefined);
  });

  test("rejects values that are not strings", function () {
    assert.strictEqual(parseDAPEnv({ FOO: 1 }), undefined);
    assert.strictEqual(parseDAPEnv(["FOO=1", 1]), undefined);
    assert.strictEqual(parseDAPEnv("FOO=1"), undefined);
  });
});

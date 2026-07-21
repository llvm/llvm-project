import * as assert from "assert";

import { convertToInteger } from "../../src/commands/pid-helpers";

suite("convertToInteger", function () {
  test("passes through integers", function () {
    assert.strictEqual(convertToInteger(0), 0);
    assert.strictEqual(convertToInteger(42), 42);
    assert.strictEqual(convertToInteger(-1), -1);
  });

  test("rejects non-integer numbers", function () {
    assert.strictEqual(convertToInteger(1.5), undefined);
    assert.strictEqual(convertToInteger(Number.NaN), undefined);
    assert.strictEqual(convertToInteger(Number.POSITIVE_INFINITY), undefined);
  });

  test("parses integer strings", function () {
    assert.strictEqual(convertToInteger("123"), 123);
    assert.strictEqual(convertToInteger("0"), 0);
    assert.strictEqual(convertToInteger("-7"), -7);
  });

  test("rejects non-integer strings", function () {
    assert.strictEqual(convertToInteger("abc"), undefined);
    assert.strictEqual(convertToInteger("12abc"), undefined);
    assert.strictEqual(convertToInteger("1.5"), undefined);
  });

  test("rejects non-number non-string values", function () {
    assert.strictEqual(convertToInteger(undefined), undefined);
    assert.strictEqual(convertToInteger(null), undefined);
    assert.strictEqual(convertToInteger(true), undefined);
    assert.strictEqual(convertToInteger({}), undefined);
    assert.strictEqual(convertToInteger([1]), undefined);
  });
});

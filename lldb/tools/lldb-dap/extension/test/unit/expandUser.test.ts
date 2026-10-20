import * as assert from "assert";
import * as os from "os";
import * as process from "process";
import { expandUser } from "../../src/utils";

suite("expandUser Test", function () {
  const home_env: { [key: string]: string | undefined } = {};
  const local_username = os.userInfo().username;

  suiteSetup(function () {
    if (os.platform() === "win32") {
      this.skip();
    }
    home_env.HOME = process.env.HOME;
    process.env.HOME = "/home/buildbot";
  });

  suiteTeardown(function () {
    process.env.HOME = home_env.HOME;
  });

  test("tilde ", function () {
    assert.equal(expandUser("~"), "/home/buildbot");
    assert.equal(expandUser("~/"), "/home/buildbot/");
    assert.equal(expandUser("~/worker"), "/home/buildbot/worker");
  });

  test("tilde with username", function () {
    assert.equal(expandUser(`~${local_username}`), "/home/buildbot");
    assert.equal(expandUser(`~${local_username}/`), "/home/buildbot/");
    assert.equal(expandUser(`~${local_username}/dev`), "/home/buildbot/dev");

    // test unknown user
    assert.notEqual(expandUser("~not_a_user"), "/home/build/bot");
  });

  test("empty", function () {
    assert.equal(expandUser(""), "");
  });

  test("no tilde", function () {
    assert.equal(expandUser("/home/buildbot/worker"), "/home/buildbot/worker");
  });
});

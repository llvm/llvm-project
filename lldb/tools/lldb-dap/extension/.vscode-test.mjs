import { defineConfig } from "@vscode/test-cli";

export default defineConfig([
  {
    label: "unitTests",
    files: "out/test/unit/**/*.test.js",
  },
]);

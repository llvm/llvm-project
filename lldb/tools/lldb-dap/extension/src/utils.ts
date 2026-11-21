import * as os from "os";
import * as path from "path";

const ENV_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const ENV_ENTRY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*(=[\s\S]*)?$/;

/**
 * Expands the character `~` to the user's home directory
 */
export function expandUser(file_path: string): string {
  if (os.platform() === "win32") {
    return file_path;
  }

  if (!file_path) {
    return "";
  }

  if (!file_path.startsWith("~")) {
    return file_path;
  }

  const path_len = file_path.length;
  if (path_len === 1) {
    return os.homedir();
  }

  if (file_path.charAt(1) === path.sep) {
    return path.join(os.homedir(), file_path.substring(1));
  }

  const sep_index = file_path.indexOf(path.sep);
  const user_name_end = sep_index === -1 ? file_path.length : sep_index;
  const user_name = file_path.substring(1, user_name_end);
  try {
    if (user_name === os.userInfo().username) {
      return path.join(os.homedir(), file_path.substring(user_name_end));
    }
  } catch (error) {
    return file_path;
  }

  return file_path;
}

/**
 * Parses the DAP environment provided in the debug configuration.
 * It must be a dictionary of string keys and values OR an array of string values.
 *
 * @param configEnv The supposed DAP environment that will be parsed
 * @returns The environment as a dictionary, or undefined if it is not valid.
 */
export function parseDAPEnv(
  configEnv: unknown,
): Record<string, string> | undefined {
  // The env must be an array or an object.
  if (typeof configEnv !== "object" || configEnv === null) {
    return undefined;
  }

  const env: Record<string, string> = {};

  // Array form: ["KEY=value", ...]
  // An array is also an object, so check for the array form first.
  if (Array.isArray(configEnv)) {
    for (const entry of configEnv) {
      if (typeof entry !== "string" || !ENV_ENTRY_PATTERN.test(entry)) {
        return undefined;
      }

      const index = entry.indexOf("=");
      if (index >= 0) {
        const key = entry.slice(0, index);
        const value = entry.slice(index + 1);
        env[key] = value;
      } else {
        env[entry] = "";
      }
    }
    return env;
  }

  // Object form: { KEY: "value", ... }
  for (const [key, value] of Object.entries(configEnv)) {
    if (typeof value !== "string" || !ENV_KEY_PATTERN.test(key)) {
      return undefined;
    }

    env[key] = value;
  }
  return env;
}

import * as os from "os";
import * as path from "path";

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

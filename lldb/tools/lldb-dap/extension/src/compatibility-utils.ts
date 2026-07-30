export function supportsCliFlag(helpText: string, flag: string): boolean {
  const escapedFlag = flag.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|\\s)${escapedFlag}(\\s|$)`, "m").test(helpText);
}

/**
 * Finds the key in `env` matching `key` case-insensitively, if any. On
 * Windows, environment variable names are case-insensitive, so a lookup for
 * "PATH" must also match "Path" or "path".
 */
export function getEnvironmentKey(
  env: { [key: string]: string },
  key: string,
): string | undefined {
  const lowerKey = key.toLowerCase();
  return Object.keys(env).find(
    (currentKey) => currentKey.toLowerCase() === lowerKey,
  );
}

/** Returns the value of `key` in `env` using a case-insensitive lookup. */
export function getEnvironmentValue(
  env: { [key: string]: string } | undefined,
  key: string,
): string | undefined {
  if (!env) {
    return undefined;
  }
  const matchedKey = getEnvironmentKey(env, key);
  return matchedKey !== undefined ? env[matchedKey] : undefined;
}

/**
 * Sets `env[key] = value`, reusing the casing of any existing key that
 * matches case-insensitively (e.g. writing "PATH" onto an existing "Path").
 */
export function setEnvironmentValue(
  env: { [key: string]: string },
  key: string,
  value: string,
): void {
  const existingKey = getEnvironmentKey(env, key);
  if (existingKey) {
    env[existingKey] = value;
  } else {
    env[key] = value;
  }
}

/**
 * Copies `overrides` onto `target`. On Windows, environment variable names
 * are case-insensitive, so each override is written through
 * {@link setEnvironmentValue} to avoid creating a second, differently-cased
 * key (e.g. both "Path" and "PATH") that would leave the override ignored.
 */
export function applyEnvironmentOverrides(
  target: { [key: string]: string },
  overrides: { [key: string]: string },
  platform: NodeJS.Platform = process.platform,
): void {
  for (const [key, value] of Object.entries(overrides)) {
    if (platform === "win32") {
      setEnvironmentValue(target, key, value);
    } else {
      target[key] = value;
    }
  }
}

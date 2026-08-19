/**
 * Converts the given value to an integer if it isn't already. Returns
 * `undefined` if the value is not an integer or a string that parses as one.
 */
export function convertToInteger(value: unknown): number | undefined {
  let result: number | undefined;
  switch (typeof value) {
    case "number":
      result = value;
      break;
    case "string":
      result = Number(value);
      break;
    default:
      return undefined;
  }
  return Number.isInteger(result) ? result : undefined;
}

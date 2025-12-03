import {
  RequestType,
  URI,
  Position,
  Range,
  uinteger,
} from 'vscode-languageclient';


/* CFG-related messages */

export namespace LlvmGetCfg {
  export interface Params {
    uri: URI;
    position: Position;
  }
  export interface Response {
    uri: URI;
    node_id: string;
    function: string;
  }
  export const Type = new RequestType<Params, Response, void>('llvm/getCfg');
}

export namespace LlvmBbLocation {
  export interface Params {
    uri: URI;
    node_id: string;
  }
  export interface Response {
    uri: URI;
    range: Range;
  }
  export const Type = new RequestType<Params, Response, void>('llvm/bbLocation');
}
